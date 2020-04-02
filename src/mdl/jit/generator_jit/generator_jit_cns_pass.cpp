/******************************************************************************
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

#include <vector>
#include <map>
#include <set>
#include <stack>
#include <cstdio>

#include <llvm/ADT/ilist_node.h>
#include <llvm/ADT/GraphTraits.h>
#include <llvm/IR/Dominators.h>
#include <llvm/Support/GenericDomTree.h>
#include <llvm/Support/GenericDomTreeConstruction.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/Local.h>
#include <llvm/Transforms/Utils/ValueMapper.h>

#include "generator_jit_cns_pass.h"

// #define DUMP_LIMITGRAPHS

// This pass implements "Making Graphs Reducible with Controlled Node Splitting"
// by Johan Janssen and Henk Corporaal.

namespace llvm {
namespace hlsl {
// forward
class CNSFunction;
class Node;

// A node in the CNS Function.
class Node : public ilist_node_with_parent<Node, CNSFunction> {
    friend class CNSLimitGraph;
    friend class CNSFunction;

public:
    typedef std::vector<Node *>       NodeList;
    typedef std::vector<BasicBlock *> BB_list;

    /// Flags for SCC.
    enum Flags {
        ON_STACK  = 1 << 0,
        ON_LOOP   = 1 << 1,
        IS_HEADER = 1 << 2,
    };

public:
    /// Default constructor.
    Node()
    : m_parent(nullptr)
    , m_id(0)
    , m_copied_from(0)
    , m_visit_count(0)
    , m_dfs_num(0)
    , m_low(0)
    , m_flags(0)
    {
    }

    /// Constructor from a basic block.
    Node(CNSFunction *parent, size_t id, BasicBlock *bb)
    : m_parent(parent)
    , m_id(id)
    , m_copied_from(0)
    , m_visit_count(0)
    , m_dfs_num(0)
    , m_low(0)
    , m_flags(0)
    {
        add_bb(bb);
    }

    /// Constructor (used by copy).
    Node(CNSFunction *parent, size_t id, size_t copied_from)
    : m_parent(parent)
    , m_id(id)
    , m_copied_from(copied_from)
    , m_visit_count(0)
    , m_dfs_num(0)
    , m_low(0)
    , m_flags(0)
    {
    }

public:
    /// Get the parent.
    CNSFunction *getParent() { return m_parent; }

    /// Set the parent.
    void setParent(CNSFunction *parent) { m_parent = parent; }

    /// Get the ID of this node.
    size_t get_id() const { return m_id; }

    /// Get the ID of the "original" node this node was copied from (if != zero).
    size_t copied_from() const { return m_copied_from; }

    /// Add a basic block to the list of basic blocks.
    void add_bb(BasicBlock *BB) {
        m_blocks.push_back(BB);
    }

    /// Add a predecessor edge to node pred.
    void add_pred(Node *pred) {
        if (pred != this) {
            if (std::find(m_preds.begin(), m_preds.end(), pred) == m_preds.end())
                m_preds.push_back(pred);
        }
    }

    /// Get all predecessors.
    NodeList &preds() { return m_preds; }

    /// Get all predecessors.
    NodeList const &preds() const { return m_preds; }

    /// Get the number of predecessors.
    size_t pred_size() const { return m_preds.size(); }

    /// Get the predecessor at index i.
    Node *get_pred(size_t i) { return m_preds[i]; }

    /// Erase the predecessor at index i.
    void pred_erase(size_t i) {
        m_preds.erase(m_preds.begin() + i);
    }

    /// Erase a given predecessor.
    void pred_erase(Node *pred) {
        m_preds.erase(std::find(m_preds.begin(), m_preds.end(), pred));
    }

    /// Add a successor edge to node succ.
    void add_succ(Node *succ) {
        if (succ != this) {
            if (std::find(m_succs.begin(), m_succs.end(), succ) == m_succs.end()) {
                m_succs.push_back(succ);
            }
        }
    }

    /// Get all successors.
    NodeList &succs() { return m_succs; }

    /// Get all successors.
    NodeList const &succs() const { return m_succs; }

    /// Get the number of successors.
    size_t succ_size() const { return m_succs.size(); }

    /// Get the successor at index i.
    Node *get_succ(size_t i) const { return m_succs[i]; }

    /// Erase the successor at index i.
    void succ_erase(size_t i) {
        m_succs.erase(m_succs.begin() + i);
    }

    /// Erase a given successor.
    void succ_erase(Node *succ) {
        m_succs.erase(std::find(m_succs.begin(), m_succs.end(), succ));
    }

    /// Get an iterator to the first block of this node.
    BB_list::iterator blocks_begin() { return m_blocks.begin(); }

    /// Get an iterator to the end block of this node.
    BB_list::iterator blocks_end()   { return m_blocks.end(); }

    /// Get the basic blocks.
    BB_list &blocks() { return m_blocks; }

    /// Mark a node on the stack.
    void markOnStack(bool flag) { if (flag) m_flags |= ON_STACK; else m_flags &= ~ON_STACK; }

    /// Check if we are inside the stack.
    bool onStack() const { return (m_flags & ON_STACK) != 0; }

    /// mark a node inside a loop.
    void markLoop(bool flag) { if (flag) m_flags |= ON_LOOP; else m_flags &= ~ON_LOOP; }

    /// Check if we are inside a loop.
    bool is_loop() const { return (m_flags & ON_LOOP) != 0; }

    /// Mark a node inside a loop as a header.
    void mark_loop_head(bool flag) { if (flag) m_flags |= IS_HEADER; else m_flags &= ~IS_HEADER; }

    /// Check if this node is a loop header.
    bool is_loop_head() const { return (m_flags & IS_HEADER) != 0; }

    void printAsOperand(raw_ostream &OS, bool flag) {}

private:
    /// The (function) parent.
    CNSFunction            *m_parent;

    /// Unique ID of this node, != zero.
    size_t                 m_id;

    /// If != zero, the ID of the original node that was copied into this one.
    size_t                 m_copied_from;

    /// Visit count for traversal.
    size_t                 m_visit_count;

    /// Predecessor edges (dependency).
    NodeList              m_preds;

    /// Successor edges (control flow).
    NodeList              m_succs;

    /// Set of basic blocks in this node.
    BB_list               m_blocks;

    // ---------  SCC helper --------------

    /// DFS number this node was reached first.
    size_t                 m_dfs_num;

    /// Minimum DFS number over this node and successors.
    size_t                 m_low;

    /// Node flags
    unsigned               m_flags;
};

// The dominator tree over the limit graph.
typedef DomTreeBase<Node> LGDominatorTree;

class CNSFunction {
public:
    typedef std::list<Node *> NodeList;

public:
    CNSFunction(Function &func)
    : m_func(func)
    , m_node_id(0)
    {
        // create a Block node for every basic block
        for (BasicBlock &BB : m_func) {
            createNode(&BB);
        }

        // now connect them
        for (BasicBlock &BB : m_func) {
            Node *curr = m_mapping[&BB];

            for (BasicBlock *p_bb : predecessors(&BB)) {
                Node *pred = m_mapping[p_bb];
                curr->add_pred(pred);
            }
            for (BasicBlock *s_bb : successors(&BB)) {
                Node *succ = m_mapping[s_bb];
                curr->add_succ(succ);
            }
        }

        // the start block
        m_root = m_mapping[&*m_func.begin()];
    }

    ~CNSFunction() {
        while (!m_node_list.empty()) {
            Node *n = m_node_list.front();
            m_node_list.pop_front();
            delete n;
        }
    }

    NodeList::iterator       begin()       { return m_node_list.begin(); }
    NodeList::const_iterator begin() const { return m_node_list.begin(); }

    NodeList::iterator       end()       { return m_node_list.end(); }
    NodeList::const_iterator end() const { return m_node_list.end(); }

    Node       *front()       { return *m_node_list.begin(); }
    Node const *front() const { return *m_node_list.begin(); }

    size_t size() const { return m_node_list.size(); }

    Function &getFunction() {
        return m_func;
    }

    StringRef getName() {
        return m_func.getName();
    }

    /// Create a new node in the graph.
    Node *createNode(BasicBlock *bb);

    /// Create a new StmtNode as a copy of another StmtNode.
    Node *createNode(Node const *other);

    /// Delete a node.
    void dropNode(Node *n);

    /// Update the mapping of a basic block to a StmtNode.
    void updateMapping(BasicBlock *bb, Node *new_node) {
        m_mapping[bb] = new_node;
    }

private:
    Function &m_func;

    /// last used node ID.
    size_t m_node_id;

    /// The top level node.
    Node *m_root;

    std::map<BasicBlock *, Node *> m_mapping;

    /// The list of all statement nodes.
    NodeList m_node_list;
};

/// A Limit Graph.
class CNSLimitGraph
{
    /// A backedge.
    struct BackEdge {
        BackEdge(Node const *S, Node const *D) : src(S), dst(D) {}

        Node const *src;
        Node const *dst;
    };

    /// Less operator for Backedges.
    struct EdgeLess {
        bool operator()(BackEdge const &a, BackEdge const &b) const {
            if (a.src < b.src) return true;
            if (a.src == b.src) return a.dst < b.dst;
            return false;
        }
    };

    typedef std::set<BackEdge, EdgeLess> BackEdgeSet;

public:
    typedef std::list<Node *> WorkList;
    typedef std::set<Node *>  NodeSet;

public:
    /// Constructor.
    CNSLimitGraph(
        CNSFunction &func);

    /// Destructor.
    ~CNSLimitGraph();

    /// remove irreducible control flow using Controlled Node Splitting.
    bool removeIrreducibleControlFlow();

    WorkList::iterator nodes_begin() {
        return m_work_list.begin();
    }

    WorkList::iterator nodes_end() {
        return m_work_list.end();
    }

    size_t nodes_size() const {
        return m_work_list.size();
    }

    /// get the start node of the limit graph.
    Node *get_start() const {
        return m_start;
    }

private:
    /// Build the limit graph from a function.
    void buildFromFunction();

    /// Apply T2 transformations until limit graph is reached, T1 is done "automatically".
    void applyT2();

    /// Recompile the dominance info for the limit graph.
    void recalculateDominance(CNSFunction &func);

    /// helper for SCC calculation.
    void doDFS(Node *node);

    /// Computes the split node candidate set using the CNS approach.
    NodeSet computeCNSCandidates();

    typedef std::set<Node *> SCC;

    /// process a strongly connected component.
    Node *processSCC(SCC const &scc);

    /// Calculate all SED-sets from one loop.
    void calcSEDsForLoop(SCC const &scc);

    /// Dump the current graph.
    void dumpLimitGraph(std::string const &baseName, size_t dumpID);

    /// Check if the edge from S->D is a backedge.
    bool isBackedge(Node const *S, Node const *D) {
        return m_backedges.find(BackEdge(S, D)) != m_backedges.end();
    }

    /// Mark the edge from S->D as a backedge.
    void mark_backedge(Node *S, Node *D) {
        m_backedges.insert(BackEdge(S, D));
    }

private:
    /// Used for scc calculation.
    size_t m_next_dfs_num;

    /// current visit count
    size_t m_visit_count;

    /// The node stack.
    std::stack<Node *> m_stack;

    /// The processed function.
    CNSFunction &m_func;

    /// The current work list.
    WorkList m_work_list;

    /// The set of backedges.
    BackEdgeSet   m_backedges;

    Node *m_start;

    // ---------  Dominance --------------
    LGDominatorTree *m_DT;

    // ---------  SED-sets --------------
    typedef std::vector<Node *> NodeVector;

    typedef std::vector<NodeVector> CandidateSEDs;
    CandidateSEDs  m_candidate_sets;

    typedef std::vector<SCC> Loops;
    Loops          m_all_loops;
};

}  // namespace hlsl

template <>
struct GraphTraits<hlsl::Node *> {
    typedef hlsl::Node                     NodeType;
    typedef hlsl::Node                     *NodeRef;
    typedef hlsl::Node::NodeList::iterator ChildIteratorType;

    static NodeType *getEntryNode(NodeType *N) { return N; }

    static inline ChildIteratorType child_begin(NodeType *N) {
        return N->succs().begin();
    }
    static inline ChildIteratorType child_end(NodeType *N) {
        return N->succs().end();
    }
};

template <>
struct GraphTraits<Inverse<hlsl::Node *> > {
    typedef hlsl::Node                     NodeType;
    typedef hlsl::Node                     *NodeRef;
    typedef hlsl::Node::NodeList::iterator ChildIteratorType;

    static NodeType *getEntryNode(NodeType *N) { return N; }

    static inline ChildIteratorType child_begin(NodeType *N) {
        return N->preds().begin();
    }
    static inline ChildIteratorType child_end(NodeType *N) {
        return N->preds().end();
    }
};

template<>
struct GraphTraits<hlsl::CNSFunction *> : public GraphTraits<hlsl::Node *> {
    typedef hlsl::CNSFunction                     GraphType;
    typedef hlsl::Node                        NodeType;
    typedef hlsl::CNSFunction::NodeList::iterator ChildIteratorType;

    //    Return the entry node of the graph
    static NodeType *getEntryNode(GraphType *G) { return G->front(); }

    //    nodes_iterator/begin/end - Allow iteration over all nodes in the graph
    typedef hlsl::CNSFunction::NodeList::iterator nodes_iterator;

    static nodes_iterator nodes_begin(GraphType *G) {
        return G->begin();
    }

    static nodes_iterator nodes_end(GraphType *G) {
        return G->end();
    }

    //    Return total number of nodes in the graph
    static size_t size(GraphType *G) {
        return G->size();
    }
};

namespace hlsl {

// -----------------------------------------------------------------------------------------------

// Constructor.
CNSLimitGraph::CNSLimitGraph(
    CNSFunction &func)
: m_next_dfs_num(0)
, m_visit_count(0)
, m_func(func)
, m_start(NULL)
{
    m_DT = new LGDominatorTree;
}

// Destructor.
CNSLimitGraph::~CNSLimitGraph()
{
    delete m_DT;
}

// Create a new node in the graph.
Node *CNSFunction::createNode(BasicBlock *BB)
{
    Node *n = new Node(this, ++m_node_id, BB);
    m_node_list.push_back(n);
    m_mapping[BB] = n;

    return n;
}

// Create a new StmtNode as a copy of another StmtNode.
Node *CNSFunction::createNode(Node const *other)
{
    size_t copied = other->m_copied_from;
    if (copied == 0)
        copied = other->m_id;

    Node *n = new Node(this, ++m_node_id, copied);
    m_node_list.push_back(n);
    return n;
}

// Delete a node.
void CNSFunction::dropNode(Node *n)
{
    /*
    new (n) StmtNode();
    m_child_list.push_back(n);
    */
}

// Build the limit graph from a function.
void CNSLimitGraph::buildFromFunction()
{
    for (Node *node : m_func) {
        m_work_list.push_back(node);
    }

    // the start block
    m_start = m_func.front();
}

void CNSLimitGraph::applyT2()
{
    for (;;) {
        // dbgs() << "Nodes in graph: " << m_work_list.size() << '\n';
        size_t removed = 0;

        for (WorkList::iterator it = m_work_list.begin(), end(m_work_list.end()); it != end;) {
            Node *curr = *it;
            // T2: Merge node into single predecessor
            if (curr->pred_size() == 1) {
                Node *pred = curr->get_pred(0);
                // dbgs() << "  - " << pred->id << " -> " << cur->id << '\n';
                pred->succ_erase(curr);

                for (BasicBlock *BB : curr->blocks()) {
                    pred->add_bb(BB);

                    // update mapping to one valid node (BB may be part of multiple nodes!)
                    m_func.updateMapping(BB, pred);
                }

                for (Node *succ : curr->succs()) {
                    // dbgs() <<  "  + " << pred->id << " -> " << succ->id << '\n';
                    succ->pred_erase(curr);
                    succ->add_pred(pred);
                    pred->add_succ(succ);
                }

                ++removed;
                it = m_work_list.erase(it);
                m_func.dropNode(curr);
            } else {
                ++it;
            }
        }

        // dbgs() << "removed: " << removed << '\n';
        if (removed == 0)
            break;
    }
    // dbgs() << "Remaining: " << m_work_list.size() << '\n';
    // dbgs() << "----------\n";
}

// Recompile the dominance info for the limit graph.
void CNSLimitGraph::recalculateDominance(CNSFunction &func)
{
    m_DT->recalculate(func);
}

// helper for SCC calculation.
void CNSLimitGraph::doDFS(Node *node)
{
    if (m_visit_count <= node->m_visit_count)
        return;
    node->m_visit_count = m_visit_count;

    node->m_dfs_num = m_next_dfs_num++;
    node->m_low = node->m_dfs_num;

    m_stack.push(node);
    node->markOnStack(true);

    for (Node *succ : node->succs()) {
        if (isBackedge(node, succ))
            continue;

        // visit the successor
        doDFS(succ);
        if (succ->onStack()) {
            node->m_low = std::min(succ->m_low, node->m_low);
        }
    }

    if (node->m_low == node->m_dfs_num) {
        Node *x = NULL;

        // found SCC
        SCC scc;
        do {
            x = m_stack.top();
            m_stack.pop();
            scc.insert(x);
        } while (x != node);

        size_t size = scc.size();
        if (size == 1) {
            // single entry scc, no loop (keep in mind we ignore self loop edges)
            x = *scc.begin();
            x->markLoop(false);
            x->markOnStack(false);
        } else {
            // found a scc of more than one node
            Node *head = processSCC(scc);

            if (head != NULL) {
                // search for inner loops
                doDFS(head);
            }
        }
    }
}

// Calculate the strongly coupled components.
CNSLimitGraph::NodeSet CNSLimitGraph::computeCNSCandidates()
{
    // clear the candidate sets
    m_candidate_sets.clear();
    m_all_loops.clear();

    ++m_visit_count;
    m_next_dfs_num = 0;

    m_backedges.clear();

    doDFS(m_start);

#if 0
    for (auto &s : m_candidate_sets) {
        dbgs() << "SET [";
        for (Node *n : s) {
            dbgs() << " " << n->get_id();
        }
        dbgs() << " ]\n";
    }
#endif

    NodeSet candidates;

    // put all nodes in the SED-sets into the candidate set
    for (auto &s : m_candidate_sets) {
        candidates.insert(s.begin(), s.end());
    }

    NodeSet result(candidates);

    for (Node *n : candidates) {
        Node *idom = m_DT->getNode(n)->getIDom()->getBlock();

        if (result.find(idom) != result.end()) {
            // idom is still a valid candidate, search all loops
            // if idom is in the same loop as n, then idom is an RC
            // and must be removed
            for (auto &loop : m_all_loops) {
                if (loop.find(n) == loop.end()) {
                    continue;
                }
                if (loop.find(idom) == loop.end()) {
                    continue;
                }

                result.erase(idom);
                break;
            }
        }
    }

#if 0
    dbgs() << "Candidates for splitting are: [";
    for (Node *n : result) {
        dbgs() << " " << n->get_id();
    }
    dbgs() << " ]\n";
#endif

    return result;
}

// process a strongly connected component.
Node *CNSLimitGraph::processSCC(SCC const &scc)
{
#if 0
    dbgs() << "Loop";
    for (Node *node : scc) {
        dbgs() << " " << unsigned(node->get_id());
    }
    dbgs() << "\n";
#endif

    Node *head = NULL;
    Node *entry = NULL;
    for (Node *node : scc) {
        node->markLoop(true);

        bool some_outside = false, some_inside = false;

        size_t max_dfs_num = 0;
        Node *max_P = NULL;

        for (Node *P : node->preds()) {
            if (isBackedge(P, node))
                continue;

            if (scc.find(P) == scc.end()) {
                some_outside = true;
            } else {
                some_inside = true;

                size_t dfs_num = P->m_dfs_num;
                if (dfs_num > max_dfs_num) {
                    max_dfs_num = dfs_num;
                    max_P = P;
                }
            }
        }
        if (some_inside && some_outside) {
            // found one successor outside the loop, this is a head
            node->mark_loop_head(true);

#if 0
            dbgs() << "Head is " << node->get_id() << "\n";
            dbgs() << "Backedge " << max_P->get_id() << "==>" << node->get_id() << "\n";
#endif
            mark_backedge(max_P, node);

            head = node;
            entry = max_P;
            break;
        }
    }

    // clean up and mark
    for (Node *node : scc) {
        node->markLoop(true);
        node->markOnStack(false);

        // set the visit count to zero here ensures that nodes on this loop are
        // visited again to find inner loops
        node->m_visit_count = 0;
    }

    calcSEDsForLoop(scc);

    return head;
}

// Calculate all SED-sets from one loop.
void CNSLimitGraph::calcSEDsForLoop(SCC const &scc)
{
    typedef std::map<Node *, NodeVector> SED_sets;

    SED_sets sets;

    for (Node *n : scc) {
        Node *idom = m_DT->getNode(n)->getIDom()->getBlock();

        if (scc.find(idom) == scc.end()) {
            // this block has a dominator outside of this loop
            if (sets.find(idom) == sets.end()) {
                // first hit
                NodeVector s;
                sets[idom] = s;
            }
            sets[idom].push_back(n);
        }
    }

    // safe SED-sets ...
    for (auto v : sets) {
        m_candidate_sets.push_back(v.second);
    }

    // ... and loops
    m_all_loops.push_back(scc);
}


bool CNSLimitGraph::removeIrreducibleControlFlow()
{
    size_t numBlocks = 0;
    size_t numInsts = 0;

    bool changed = false;

    // test reducibility
#if 0
    dbgs() << "Computing limit graph for " << m_func.getName() << "\n";
    dbgs() << "------------------------------------\n";
#endif
    buildFromFunction();

    size_t dumpID = 0;

    for (;;) {
        dumpLimitGraph(m_func.getName(), dumpID++);

        applyT2();
        if (m_work_list.size() == 1) {
            // reducible
            break;
        }

        changed = true;

        // for now, we recalculate the dominance at every run
        //  it *might* be possible to update it ...
        recalculateDominance(m_func);

        NodeSet candidates = computeCNSCandidates();

        dumpLimitGraph(m_func.getName(), dumpID++);

        // Select node for splitting
        Node *best = NULL;
        size_t bestNum = 1 << 30;
        // dbgs() << "Candidates:\n";
        for (Node *N : candidates) {
            size_t num = 0;
            for (BasicBlock *B : N->blocks())
                num += B->size();
#if 0
            dbgs() << "  " << N->get_id()
                   << " : " << num
                   << " * " << (N->pred_size() - 1)
                   << " = " << num * (N->pred_size() - 1) <<  "\n";
#endif
            num *= (N->pred_size() - 1);
            if (num < bestNum) {
                best = N;
                bestNum = num;
            }
        }

        // Split block
        m_work_list.remove(best);

#if 0
        dbgs() << "Splitting id = " << best->get_id()
               << " into " << best->pred_size()
               << " copies (" << bestNum
               << " more instructions and " << best->blocks().size() * (best->pred_size() - 1)
            << " more blocks)\n";
#endif
        numBlocks += best->blocks().size() * (best->pred_size() - 1);
        numInsts += bestNum;

        for (Node *S : best->succs()) {
            S->pred_erase(best);
        }

        // Turn values which escape their blocks into variables to avoid problems during cloning
        std::set<Instruction *> to_demote;
        for (BasicBlock *BB : best->blocks()) {
            for (Instruction &inst : *BB) {
                // A value escapes when a use is in another block or is a phi node
                for (Instruction::use_iterator UI = inst.use_begin(), UE = inst.use_end();
                        UI != UE; ++UI) {
                    Instruction* useI = cast<Instruction>(*UI);
                    if (useI->getParent() != BB || isa<PHINode>(useI)) {
                        // dbgs() << "Demoting value: " << inst << '\n';
                        to_demote.insert(&inst);
                        break;
                    }
                }
            }
        }

        for (Instruction* inst : to_demote) {
            DemoteRegToStack(*inst, /*VolatileLoads=*/false);
        }

        // Create one clone of the best node for each predecessor
        for (Node *P : best->preds()) {
            // Create a new node with P as only predecessor and all original successors
            // dbgs() << "Creating clone for id: " << P->get_id() << '\n';
            Node *newBlock = m_func.createNode(best);
            newBlock->add_pred(P);
            newBlock->succs() = best->succs();

            m_work_list.push_back(newBlock);

            // Update adjacent nodes
            P->succ_erase(best);
            P->add_succ(newBlock);
            for (Node *S : best->succs()) {
                S->add_pred(newBlock);
            }

            // Clone all basic blocks
            ValueToValueMapTy valueMap;
            for (BasicBlock *BB : best->blocks()) {
                // dbgs() << "Clone block: " << BB->getName() << '\n';
                BasicBlock* NB = CloneBasicBlock(BB, valueMap, "", &m_func.getFunction());
                valueMap[BB] = NB;
                newBlock->add_bb(NB);
            }

            // Remap operands of instructions in cloned blocks to cloned values, if available
            for (BasicBlock *BB : best->blocks()) {
                BasicBlock *clonedBlock = cast<BasicBlock>(valueMap[BB]);
                for (Instruction &Instr : *clonedBlock) {
                    RemapInstruction(
                        &Instr,
                        valueMap,
                        RF_NoModuleLevelChanges |
                        RF_IgnoreMissingLocals |
                        RF_NullMapMissingGlobalValues);
                }
            }

            // Set the successors of the blocks in the predecessor node to the cloned blocks
            // and collect the blocks where we potentially need to update phis
            std::set<BasicBlock*> phiUpdateBlocks;
            // dbgs() << "Looking at " << P->blocks().size() << " predecessor blocks\n";
            for (BasicBlock *BB : P->blocks()) {
                TerminatorInst *term = BB->getTerminator();
                unsigned numSuccessors = term->getNumSuccessors();

                for (unsigned idx = 0; idx < numSuccessors; ++idx) {
                    BasicBlock* successor = term->getSuccessor(idx);
                    Value *v = valueMap[successor];
                    if (BasicBlock *clonedSuccessor = cast_or_null<BasicBlock>(v)) {
                        term->setSuccessor(idx, clonedSuccessor);
                        // dbgs() << "  updated terminator: " << *term << "\n";

                        phiUpdateBlocks.insert(clonedSuccessor);
                    }
                }
            }
            if (phiUpdateBlocks.empty()) {
                dbgs() << "FAIL\n";
                abort();
            }

            // Cloning the blocks added new predecessors to non-cloned blocks without
            // updating the phis. Do it now
            for (BasicBlock *origBlock : best->blocks()) {
                BasicBlock *clonedBlock = cast<BasicBlock>(valueMap[origBlock]);
                for (auto BI = succ_begin(origBlock), BE = succ_end(origBlock); BI != BE; ++BI) {
                    if (valueMap[*BI] == NULL) {
                        // the successor of the original block has not been cloned,
                        // so the clone is a new predecessor
                        for (auto II = BI->begin(), IE = BI->end(); II != IE; ++II) {
                            if (PHINode *phi = dyn_cast<PHINode>(II)) {
                                int predIndex = phi->getBasicBlockIndex(origBlock);

                                // use the cloned version of the phi value, if available,
                                // the original otherwise
                                Value *origPhiValue = phi->getIncomingValue(predIndex);
                                Value *newPhiValue = valueMap[origPhiValue];
                                if (newPhiValue == NULL)
                                    newPhiValue = origPhiValue;

                                phi->addIncoming(newPhiValue, clonedBlock);
                            } else
                                break;  // no more phi nodes in the block
                        }
                    }
                }
            }

            // Remove phi operands in cloned blocks for non-existing predecessors
            for (BasicBlock *block : phiUpdateBlocks) {
                PHINode *firstPHI = dyn_cast<PHINode>(block->begin());
                if (firstPHI == nullptr)
                    continue;

                std::set<BasicBlock *> preds(pred_begin(block), pred_end(block));
                if (preds.size() == firstPHI->getNumIncomingValues())
                    continue;

                for (PHINode &phi : block->phis()) {
                    SmallVector <unsigned, 8> toRemove;
                    for (unsigned op = 0, opEnd = phi.getNumIncomingValues(); op != opEnd; ++op) {
                        BasicBlock *pred = phi.getIncomingBlock(op);
                        if (preds.count(pred) == 0)
                            toRemove.push_back(op);
                    }
                    for (auto I = toRemove.rbegin(), E = toRemove.rend(); I != E; ++I)
                        phi.removeIncomingValue(*I, false);
                }
            }
        }
        for (BasicBlock *BB : best->blocks()) {
            BB->setName("dead");
            // dbgs() << "ERASE: " << *BB << '\n';
            //BB->eraseFromParent();
        }
        m_func.dropNode(best);
    }

    // dbgs() << "Added: blocks=" << numBlocks << ", insts=" << numInsts << '\n';

    return changed;
}

ControlledNodeSplittingPass::ControlledNodeSplittingPass()
    : FunctionPass(ID)
{
}

bool ControlledNodeSplittingPass::runOnFunction(Function &function)
{
    CNSFunction astFunction(function);

    CNSLimitGraph LG(astFunction);

    return LG.removeIrreducibleControlFlow();
}

char ControlledNodeSplittingPass::ID = 0;

//------------------------------------------------------------------------------
Pass* createControlledNodeSplittingPass()
{
    return new ControlledNodeSplittingPass();
}

void CNSLimitGraph::dumpLimitGraph(std::string const &baseName, size_t dumpID)
{
#ifdef DUMP_LIMITGRAPHS
    std::string filename = baseName + "-cns-" + std::to_string(dumpID) + ".gv";
    FILE *file = fopen(filename.c_str(), "wt");

    fprintf(file, "digraph \"%s-%u\" {\n", baseName.c_str(), unsigned(dumpID));

    for (Node const *node : m_work_list) {
#ifdef DUMP_BLOCKNAMES
        fprintf(file, "n%u [label=\"%s",
            unsigned(node->get_id()), std::string(node->blocks()[0]->getName()).c_str());
#else
        fprintf(file, "n%u [label=\"%u",
            unsigned(node->get_id()), unsigned(node->get_id()));
#endif
        if (size_t copied_from = node->copied_from())
            fprintf(file, " (%u)", unsigned(copied_from));
        fprintf(file, "\"");
        if (node->is_loop_head())
            fprintf(file, " shape=box");
        else if (node->is_loop())
            fprintf(file, " shape=diamond");
        fprintf(file, " ]\n");
    }

    for (Node const *node : m_work_list) {
        unsigned node_id = unsigned(node->get_id());

        for (Node const *succ : node->succs()) {
            fprintf(file, "n%u -> n%u", node_id, unsigned(succ->get_id()));

            if (isBackedge(node, succ))
                fprintf(file, " [ style=dashed ]");

            fprintf(file, "\n");
        }
    }

    fprintf(file, "}\n");

    fclose(file);
#endif
}

}  // namespace hlsl
}  // namespace llvm
