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

// The algorithms in this file are based on
// Simon Moll, "Decompilation of LLVM IR", Bachelor Thesis, Saarland University, 2011
// http://www.cdl.uni-saarland.de/publications/theses/moll_bsc.pdf

#include "pch.h"

#include <algorithm>
#include <vector>
#include <map>
#include <set>
#include <stack>
#include <cstdio>

#include <llvm/ADT/ilist_node.h>
#include <llvm/ADT/GraphTraits.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/IR/Dominators.h>
#include <llvm/Support/GenericDomTree.h>
#include <llvm/Support/GenericDomTreeConstruction.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/Local.h>
#include <llvm/Transforms/Utils/ValueMapper.h>

#include "mdl/compiler/compilercore/compilercore_assert.h"

#include "generator_jit_hlsl_function.h"
#include "generator_jit_ast_compute.h"
#include "generator_jit_type_map.h"

#define DUMP_REGIONGRAPHS
#define DUMP_STMTKINDS

namespace llvm {
namespace hlsl {

// The dominator tree over the limit graph.
typedef DomTreeBase<Region> LGDominatorTree;

class ASTFunction;
class RegionContext;


/// A set container allowing to iterate over the elements in insertion order
/// to ensure deterministic behavior.
template <typename T>
class InsertionOrderSet
{
public:
    typedef typename std::vector<T>::iterator iterator;
    typedef typename std::vector<T>::const_iterator const_iterator;

    /// Returns a reference to the first inserted element of the set.
    T &front() { return m_vector.front(); }

    /// Returns a reference to the first inserted element of the set.
    T &back() { return m_vector.back(); }

    /// Returns the begin iterator used for iterating over the elements in insertion order.
    iterator begin() { return m_vector.begin(); }

    /// Returns the const begin iterator used for iterating over the elements in insertion order.
    const_iterator begin() const { return m_vector.begin(); }

    /// Returns the end iterator used for iterating over the elements in insertion order.
    iterator end() { return m_vector.end(); }

    /// Returns the const end iterator used for iterating over the elements in insertion order.
    const_iterator end() const { return m_vector.end(); }

    /// Returns the number of (unique) elements in the set.
    size_t size() const { return m_vector.size(); }

    /// Returns true, if the set is empty.
    bool empty() const { return m_vector.empty(); }

    /// Insert the given value into the set.
    /// \returns true if the value was not in the set, yet.
    bool insert(const T &val)
    {
        if (!m_set.insert(val).second)
            return false;

        m_vector.push_back(val);
        return true;
    }

    /// Insert all elements given by the iterators into the set.
    template <class Iter>
    void insert(Iter first, Iter last)
    {
        for (; first != last; ++first)
            insert(*first);
    }

    /// Erase the given value from the set.
    /// \returns the number of removed elements.
    size_t erase(T const &val)
    {
        size_t count = m_set.erase(val);
        if (count > 0) {
            iterator it = std::find(m_vector.begin(), m_vector.end(), val);
            if (it != m_vector.end()) {
                m_vector.erase(it);
            } else {
                MDL_ASSERT(!"Key found in set but not in vector!");
            }
        }
        return count;
    }

    /// Clear the set.
    void clear()
    {
        m_set.clear();
        m_vector.clear();
    }

    /// Return true if the set contains the given value.
    bool contains(T const &val) const {
        return m_set.find(val) != m_set.end();
    }

    /// Return the lastly added element and remove it from the set.
    T pop()
    {
        MDL_ASSERT(!m_vector.empty() && "Vector may not be empty");
        T val = m_vector.back();
        m_vector.pop_back();
        m_set.erase(val);
        return val;
    }

private:
    /// The set container used to quickly check for uniqueness.
    std::set<T> m_set;

    /// The vector container used to provide the insertion ordering.
    std::vector<T> m_vector;
};

typedef InsertionOrderSet<BasicBlock *> BlockSet;


/// The region builder.
class RegionBuilder
{
    friend class IfContext;

    /// A backedge.
    struct BackEdge {
        BackEdge(Region const *S, Region const *D) : src(S), dst(D) {}

        Region const *src;
        Region const *dst;
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
    typedef std::list<Region *> WorkList;
    typedef std::set<Region *>  NodeSet;

    enum PatternRule {
        PR_NONE,
        PR_RETURN,
        PR_SEQUENCE,
        PR_IF_THEN_ELSE,
        PR_IF_THEN,
        PR_IF_THEN_JUMP,
        PR_DO_WHILE,
        PR_WHILE,
        PR_BREAK,
        PR_CONTINUE
    };

public:
    /// Constructor.
    ///
    /// \param func         the LLVM function we build a region graph for
    /// \param type_mapper  the MDL type mapper
    /// \param domTree      LLVM dominance tree for \c func
    /// \param loop_info    LLVM loop info for \c func
    RegionBuilder(
        Function             &func,
        mi::mdl::Type_mapper &type_mapper,
        DominatorTree        &domTree,
        LoopInfo             &loop_info);

    /// Build regions for a function.
    ASTFunction *buildRegions();

    WorkList::iterator nodes_begin() {
        return m_work_list.begin();
    }

    WorkList::iterator nodes_end() {
        return m_work_list.end();
    }

    size_t nodes_size() const {
        return m_work_list.size();
    }

    /// Get the name of a pattern rule.
    static char const *get_pattern_rule_name(PatternRule rule) {
        switch (rule) {
        case PR_NONE: return "none";
        case PR_RETURN: return "return";
        case PR_SEQUENCE: return "sequence";
        case PR_IF_THEN_ELSE: return "if-then-else";
        case PR_IF_THEN: return "if-then";
        case PR_IF_THEN_JUMP: return "if-then-jump";
        case PR_DO_WHILE: return "do-while";
        case PR_WHILE: return "while";
        case PR_BREAK: return "break";
        case PR_CONTINUE: return "continue";
        }
        return "unknown";
    }

private:
    /// Build the initial region graph from a function.
    void buildInitialFromFunction();

    /// Discover the region started at entry,
    ///
    /// \param[in]  ctx              current context
    /// \param[in]  processLoopBody  true if we process the body inside a loop
    /// \param[in]  entry            the basic block that is the single entry of the region
    Region *discoverRegion(
        RegionContext const &ctx,
        bool                processLoopBody,
        BasicBlock          *entry);

    /// Process a basic block that form the entry point of a region.
    ///
    /// \param[in]  ctx              current context
    /// \param[in]  processLoopBody  true if we process the body inside a loop
    /// \param[in]  bb               the basic block that is processed
    /// \param[out] region           resulting region
    ///
    /// \return the next block to process
    BasicBlock *processBB(
        RegionContext const &ctx,
        bool                processLoopBody,
        BasicBlock          *BB,
        Region              *&region);

    /// Process a basic block that form the head of a loop region.
    ///
    /// \param[in]  ctx         current context
    /// \param[in]  bb          the basic block that is processed
    /// \param[out] region      resulting region
    ///
    /// \return the next block to process
    BasicBlock *processLoop(
        RegionContext const &ctx,
        BasicBlock          *BB,
        Region              *&region);

    /// Dump the current graph.
    void dumpRegionGraph(char const *suffix=nullptr, bool with_subgraphs=false);

    /// Creates a graph node.
    unsigned dumpRegion(FILE *file, Region const *region);

    /// Dump a sub graph.
    std::vector<unsigned> dumpSubGraph(FILE *file, Region const *region);

    /// Get the kind string for a region.
    static char const *getKindString(Region const *region)
    {
        char const *kind_str = "<UNKNOWN>";
        switch (region->get_kind()) {
        case Region::SK_INVALID:           kind_str = "invalid";      break;
        case Region::SK_BLOCK:             kind_str = "block";        break;
        case Region::SK_SEQUENCE:          kind_str = "sequence";     break;
        case Region::SK_IF_THEN:           kind_str = "if-then";      break;
        case Region::SK_IF_THEN_ELSE:      kind_str = "if-then-else"; break;
        case Region::SK_NATURAL_LOOP:      kind_str = "loop";         break;
        case Region::SK_BREAK:             kind_str = "break";        break;
        case Region::SK_CONTINUE:          kind_str = "continue";     break;
        case Region::SK_RETURN:            kind_str = "return";       break;
        case Region::SK_SWITCH:            kind_str = "switch";       break;
        }
        return kind_str;
    }

private:
    /// The processed function.
    ASTFunction &m_func;

    /// The current work list.
    WorkList m_work_list;

#ifdef DUMP_REGIONGRAPHS
    /// ID used for dumping the limit graphs.
    size_t m_dump_id;

    /// Base name used for dumping the limit graphs.
    std::string m_dump_base_name;
#endif
};

/// Region context information.
class RegionContext {
public:
    /// Constructor.
    ///
    /// \param exitBlock       the current exit block if any
    /// \param parentLoop      the parent loop if any
    /// \param continueTarget  the target of a continue branch if any
    /// \param breakTarget     the target of a break branch if any
    RegionContext(
        BasicBlock *exitBlock      = nullptr,
        Loop       *parentLoop     = nullptr,
        BasicBlock *continueTarget = nullptr,
        BasicBlock *breakTarget    = nullptr)
    : m_exitBlock(exitBlock)
    , m_parentLoop(parentLoop)
    , m_continueTarget(continueTarget)
    , m_breakTarget(breakTarget)
    {
    }

    /// Constructor from an exit block and another context.
    ///
    /// \param exitBlock  the exit block
    /// \param other      the other (typically parent) context
    RegionContext(
        BasicBlock          *exitBlock,
        RegionContext const &other)
    : m_exitBlock(exitBlock)
    , m_parentLoop(other.m_parentLoop)
    , m_continueTarget(other.m_continueTarget)
    , m_breakTarget(other.m_breakTarget)
    {
    }

    /// Default copy constructor.
    RegionContext(RegionContext const &) = default;

    /// Get the exit block (has the exit edge).
    BasicBlock *getExitBlock() const { return m_exitBlock; }

    /// Get the continue target block.
    BasicBlock *getContinueTarget() const { return m_continueTarget; }

    /// Get the break target block.
    BasicBlock *getBreakTarget() const { return m_breakTarget; }

    /// The parent loop if any.
    Loop *get_parentLoop() const { return m_parentLoop; }

    /// Return the set {B_break, B_continue}.
    BlockSet getRegularExitTargets() const {
        BlockSet res;
        if (m_breakTarget != nullptr)
            res.insert(m_breakTarget);
        if (m_continueTarget != nullptr)
            res.insert(m_continueTarget);
        return res;
    }

    /// Check if BB \in {B_break, B_continue}.
    bool isRegularExitTarget(BasicBlock *BB) const {
        return BB == m_breakTarget || BB == m_continueTarget;
    }

    /// Return the set {B_break, B_continue, B_exit}
    BlockSet getAnticipatedExitTargets() const {
        BlockSet res;
        if (m_breakTarget != nullptr)
            res.insert(m_breakTarget);
        if (m_continueTarget != nullptr)
            res.insert(m_continueTarget);
        if (m_exitBlock != nullptr)
            res.insert(m_exitBlock);
        return res;
    }

    /// Set the exit block.
    ///
    /// \param BB  the single exit block
    void setExitBlock(BasicBlock *BB) {
        m_exitBlock = BB;
    }

private:
    // default copy, no assignment
    RegionContext &operator=(RegionContext const &) = delete;

private:
    /// The current region exit block if any.
    BasicBlock *m_exitBlock;

    /// The parent loop if any.
    Loop       *m_parentLoop;

    /// The current continue target block if any.
    BasicBlock *m_continueTarget;

    /// The current break target block if any.
    BasicBlock *m_breakTarget;
};

/// Helper context to build acyclic regions where the entry block has 2 predecessors.
class IfContext {
public:
    /// Creates a new IfContext instance.
    ///
    /// \param ctx     current context
    /// \param func    the current (AST) function
    /// \param BB      the entry block
    static IfContext *create(
        RegionContext const &ctx,
        ASTFunction         &func,
        BasicBlock         *BB);

    /// Check if the current if-region has a single exit block, if not, try to insert one.
    ///
    /// \param[in]  ctx        current context
    /// \param[out] exitBlock  the (single) exit block of the if region
    bool handleProperRegions(
        RegionContext const &ctx,
        BasicBlock          *&exitBlock);

    /// Create a new if region.
    ///
    /// \param builder     the region builder
    /// \param exitBlock   the single exit block (if known)
    /// \param terminator  the LLVM terminator instruction that branches to then/else
    Region *createIfRegion(
        RegionBuilder  &builder,
        BasicBlock     *exitBlock,
        TerminatorInst *terminator);

private:
    /// Constructor.
    ///
    /// \param func       the owner function
    /// \param entry      the single entry block
    /// \param exit       the single exit block (if known)
    /// \param thenCtx    the region context for the then branch
    /// \param thenBlock  the then block if any
    /// \param elseCtx    the region context for the else branch
    /// \param elseBlock  the else block if any
    IfContext(
        ASTFunction         &func,
        BasicBlock          *entry,
        BasicBlock          *exit,
        RegionContext const &thenCtx,
        BasicBlock          *thenBlock,
        RegionContext const &elseCtx,
        BasicBlock          *elseBlock)
    : m_func(func)
    , m_entry(entry)
    , m_exit(exit)
    , m_thenContext(thenCtx)
    , m_elseContext(elseCtx)
    , m_thenBlock(thenBlock)
    , m_elseBlock(elseBlock)
    {
    }

    /// Compute the exit sets of the then/else branches.
    ///
    /// \param[out] thenExitSet       the set of exits of the current then
    /// \param[out] elseExitSet       the set of exits of the current else
    /// \param[out] usedRegularExits  the set of used exits from the regular exit set
    /// \param[in]  regularExits      the regular exit set
    void computeExitSets(
        BlockSet       &thenExitSet,
        BlockSet       &elseExitSet,
        BlockSet       &usedRegularExits,
        BlockSet const &regularExits);

    /// Compute the set of all blocks that are reachable (dominated) by the given header until
    /// we reach one of the exits.
    ///
    /// \param[in] header  header block of the region
    /// \param[in] exits   set of exits of the region
    BlockSet computeDominatedRegion(llvm::BasicBlock *header, BlockSet const &exits);

    /// Convert an integer value.
    ///
    /// \param i  the value
    ConstantInt *get_constant(int i);

private:
    ASTFunction   &m_func;

    BasicBlock    *m_entry;
    BasicBlock    *m_exit;

    RegionContext m_thenContext;
    RegionContext m_elseContext;

    BasicBlock    *m_thenBlock;
    BasicBlock    *m_elseBlock;
};

/// Computes the extended dominance frontier, stopping at regular exits.
///
/// \param[in] domTree                 the dominator tree of the current function
/// \param[in] BB                      the basic block
/// \param[in] regularExitTargets      set of regular exit targets
/// \param[in] usedRegularExitTargets  if non-null, collects the used regular exits
static BlockSet computeDominanceFrontierExt(
    DominatorTree  &domTree,
    BasicBlock     *BB,
    BlockSet const &regularExitTargets,
    BlockSet       *usedRegularExitTargets = nullptr)
{
    BlockSet domFrontier;

    if (regularExitTargets.contains(BB)) {
        // the start block is a regular exit target, return empty set
        if (usedRegularExitTargets != nullptr)
            usedRegularExitTargets->insert(BB);
        return domFrontier;
    }

    // do a depth-first-walk starting from BB and ending at the dominance frontier and regular exits
    std::stack<llvm::BasicBlock*> worklist;
    BlockSet seen;
    worklist.push(BB);
    seen.insert(BB);

    do {
        BasicBlock *curBlock = worklist.top();
        worklist.pop();

        for (BasicBlock *succ : successors(curBlock)) {
            if (!seen.insert(succ))
                continue;

            // a regular exit target?
            if (regularExitTargets.contains(succ)) {
                if (usedRegularExitTargets != nullptr)
                    usedRegularExitTargets->insert(succ);
                continue;
            }

            // not dominated by BB anymore -> add to dominance frontier
            if (!domTree.dominates(BB, succ)) {
                domFrontier.insert(succ);
                continue;
            }

            // continue walking
            worklist.push(succ);
        }
    } while (!worklist.empty());

    return domFrontier;
}

/// Returns true, if the block b is reachable from the block a without using one of the regular
/// exits of the given region context.
static bool reachable(
    RegionContext const &ctx,
    BasicBlock *a,
    BasicBlock *b)
{
    if (a == b)
        return true;

    // do a depth-first-search for b starting at a
    std::stack<BasicBlock *> worklist;
    BlockSet seen;
    worklist.push(a);
    seen.insert(a);

    do {
        BasicBlock *curBlock = worklist.top();
        worklist.pop();

        for (BasicBlock *succ : successors(curBlock)) {
            if (succ == b)
                return true;

            // don't allow regular exits or blocks we have already seen
            if (!ctx.isRegularExitTarget(succ) && seen.insert(succ)) {
                worklist.push(succ);
            }
        }
    } while (!worklist.empty());

    return false;
}

}  // namespace hlsl

template <>
struct GraphTraits<hlsl::Region *> {
    typedef hlsl::Region                     NodeType;
    typedef hlsl::Region                     *NodeRef;
    typedef hlsl::Region::RegionList::iterator ChildIteratorType;

    static NodeType *getEntryNode(NodeType *N) { return N; }

    static inline ChildIteratorType child_begin(NodeType *N) {
        return N->succs().begin();
    }
    static inline ChildIteratorType child_end(NodeType *N) {
        return N->succs().end();
    }
};

template <>
struct GraphTraits<Inverse<hlsl::Region *> > {
    typedef hlsl::Region                     NodeType;
    typedef hlsl::Region                     *NodeRef;
    typedef hlsl::Region::RegionList::iterator ChildIteratorType;

    static NodeType *getEntryNode(NodeType *N) { return N; }

    static inline ChildIteratorType child_begin(NodeType *N) {
        return N->preds().begin();
    }
    static inline ChildIteratorType child_end(NodeType *N) {
        return N->preds().end();
    }
};

template<>
struct GraphTraits<hlsl::ASTFunction *> : public GraphTraits<hlsl::Region *> {
    typedef hlsl::ASTFunction                       GraphType;
    typedef hlsl::Region                            NodeType;
    typedef hlsl::ASTFunction::RegionList::iterator ChildIteratorType;

    // Return the entry node of the graph
    static NodeType *getEntryNode(GraphType *G) { return G->front(); }

    // nodes_iterator/begin/end - Allow iteration over all nodes in the graph
    typedef hlsl::ASTFunction::RegionList::iterator nodes_iterator;

    static nodes_iterator nodes_begin(GraphType *G) {
        return G->begin();
    }

    static nodes_iterator nodes_end(GraphType *G) {
        return G->end();
    }

    // Return total number of nodes in the graph
    static size_t size(GraphType *G) {
        return G->size();
    }
};

namespace hlsl {

// -----------------------------------------------------------------------------------------------

// Get the entry region of this region.
Region const *Region::getEntryRegionBlock() const {
    Region const *top_region = this;
    while (top_region->get_kind() != Region::SK_BLOCK) {
        top_region = static_cast<RegionComplex const *>(top_region)->getHead();
    }
    return top_region;
}

// Get the LLVM loop information for this node.
llvm::Loop *Region::get_loop() {
    return getParent()->getLoopInfo().getLoopFor(get_head_bb());
}


// Constructor.
ASTFunction::ASTFunction(
    Function             &func,
    mi::mdl::Type_mapper &type_mapper,
    DominatorTree        &domTree,
    LoopInfo             &loop_info)
: m_func(func)
, m_type_mapper(type_mapper)
, m_domTree(domTree)
, m_loop_info(loop_info)
, m_node_id(0)
{
    BasicBlock *entry_block = &m_func.getEntryBlock();

    // create a Block region for every basic block
    for (BasicBlock &BB : m_func) {
        // no predecessors, but not entry block? -> skip unreachable block
        if (&BB != entry_block && pred_begin(&BB) == pred_end(&BB))
            continue;

        createBasicRegion(&BB);
    }

    // now connect them
    for (BasicBlock &BB : m_func) {
        // no predecessors, but not entry block? -> skip unreachable block
        if (&BB != entry_block && pred_begin(&BB) == pred_end(&BB))
            continue;

        Region *curr = m_mapping[&BB];

        for (BasicBlock *p_bb : predecessors(&BB)) {
            Region *pred = m_mapping[p_bb];
            if (pred)  // not an unreachable block?
                curr->add_pred(pred);
        }
        for (BasicBlock *s_bb : successors(&BB)) {
            Region *succ = m_mapping[s_bb];
            curr->add_succ(succ);
        }
    }

    // initialize the body with the start block
    m_body = m_mapping[&*m_func.begin()];
}

// Destructor.
ASTFunction::~ASTFunction() {
    while (!m_region_list.empty()) {
        Region *n = m_region_list.front();
        m_region_list.pop_front();
        delete n;
    }
}

// Get the LLVM loop for a given basic block.
Loop *ASTFunction::getLoopFor(BasicBlock const *BB) const
{
    return m_loop_info.getLoopFor(BB);
}

// Returns true iff A dominates B.
bool ASTFunction::dominates(BasicBlock *A, BasicBlock *B) const
{
    return m_domTree.dominates(A, B);
}

// Get top-most region for a basic block.
Region *ASTFunction::getTopRegion(BasicBlock *bb)
{
    Region *cur_node = m_mapping[bb];

    while (RegionComplex *owner = cur_node->getOwnerRegion())
        cur_node = owner;

    return cur_node;
}

// Create a new Basic node in the graph.
Region *ASTFunction::createBasicRegion(BasicBlock *BB)
{
    Region *n = new RegionBlock(this, ++m_node_id, BB);
    m_region_list.push_back(n);
    m_mapping[BB] = n;

    return n;
}

// Create a new Invalid node in the graph.
Region *ASTFunction::createInvalidRegion(Region *head)
{
    Region *n = new RegionInvalid(this, ++m_node_id, head);
    m_region_list.push_back(n);

    return n;
}

// Create a new Sequence node in the graph.
RegionSequence *ASTFunction::createSequenceRegion(
    Region                  *head,
    ArrayRef<Region *> const &tail)
{
    RegionSequence *n = new RegionSequence(this, ++m_node_id, head, tail);
    m_region_list.push_back(n);

    return n;
}

// Create a new IfThen node in the graph.
RegionIfThen *ASTFunction::createIfThenRegion(
    Region         *head,
    Region         *then,
    TerminatorInst *terminator,
    bool           negated)
{
    RegionIfThen *n = new RegionIfThen(this, ++m_node_id, head, then, terminator, negated);
    m_region_list.push_back(n);

    return n;
}

// Create a new IfThenElse node in the graph.
RegionIfThenElse *ASTFunction::createIfThenElseRegion(
    Region         *head,
    Region         *then_node,
    Region         *else_node,
    TerminatorInst *terminator)
{
    RegionIfThenElse *n = new RegionIfThenElse(
        this, ++m_node_id, head, then_node, else_node, terminator);
    m_region_list.push_back(n);

    return n;
}

// Create a new natural loop node in the graph.
RegionNaturalLoop *ASTFunction::createNaturalLoopRegion(Region *head)
{
    RegionNaturalLoop *n = new RegionNaturalLoop(this, ++m_node_id, head);
    m_region_list.push_back(n);

    return n;
}

// Create a new Break region in the graph.
RegionBreak *ASTFunction::createBreakRegion()
{
    RegionBreak *n = new RegionBreak(this, ++m_node_id);
    m_region_list.push_back(n);

    return n;
}

// Create a new Continue region in the graph.
RegionContinue *ASTFunction::createContinueRegion()
{
    RegionContinue *n = new RegionContinue(this, ++m_node_id);
    m_region_list.push_back(n);

    return n;
}

// Create a new Return region in the graph.
RegionReturn *ASTFunction::createReturnRegion(
    Region     *head,
    ReturnInst *return_inst)
{
    RegionReturn *n = new RegionReturn(this, ++m_node_id, head, return_inst);
    m_region_list.push_back(n);

    return n;
}

// Create a new Switch node in the graph.
RegionSwitch *ASTFunction::createSwitchRegion(
    Region *head, ArrayRef<Region *> const &cases, Region *def_case)
{
    RegionSwitch *n = new RegionSwitch(this, ++m_node_id, head, cases, def_case);
    m_region_list.push_back(n);

    return n;
}

// Delete a node.
void ASTFunction::dropRegion(Region *r)
{
    m_region_list.remove(r);
    delete r;
}

// Recalculate the dominance tree for the function.
void ASTFunction::updateDomTree()
{
    m_domTree.recalculate(m_func);
}

// Constructor.
RegionBuilder::RegionBuilder(
    Function             &func,
    mi::mdl::Type_mapper &type_mapper,
    DominatorTree        &domTree,
    LoopInfo             &loop_info)
: m_func(* new ASTFunction(func, type_mapper, domTree, loop_info))
#ifdef DUMP_REGIONGRAPHS
, m_dump_id(0)
#endif
{
}

// Build the limit graph from a function.
void RegionBuilder::buildInitialFromFunction()
{
    for (Region *r : m_func) {
        m_work_list.push_back(r);
    }
}

// Process a basic block that form the head of a loop region.
BasicBlock *RegionBuilder::processLoop(
    RegionContext const &ctx,
    BasicBlock          *head,
    Region              *&region)
{
    typedef std::map<BasicBlock *, Region *> RegionMap;

    Region *bodyRegion = discoverRegion(ctx, /*processLoopBody=*/true, head);

    region = m_func.createNaturalLoopRegion(bodyRegion);

    return ctx.getBreakTarget();
}

/// Get the unique exit target of a loop.
///
/// \param[in]  loop           an LLVM loop
/// \param[out] more_then_one  true, if loop contains more then one exit targets
static BasicBlock *getLoopUniqueExitTarget(Loop &loop, bool &more_then_one)
{
    std::set<BasicBlock *> loopBBs(loop.block_begin(), loop.block_end());

    BasicBlock *target = nullptr;
    more_then_one = false;

    for (BasicBlock *block : loop.blocks()) {
        TerminatorInst *term = block->getTerminator();
        for (BasicBlock *succ : term->successors()) {
            if (loopBBs.find(succ) == loopBBs.end()) {
                // found an exit target
                if (target == nullptr) {
                    target = succ;
                } else if (target != succ) {
                    more_then_one = true;
                    break;
                }
            }
        }
    }

    return target;
}

// Process a basic block that form the entry point of a region.
BasicBlock *RegionBuilder::processBB(
    RegionContext const &ctx,
    bool                processLoopBody,
    BasicBlock          *BB,
    Region              *&region)
{
    if (!processLoopBody) {
        if (BB == ctx.getBreakTarget()) {
            region = m_func.createBreakRegion();

            // do NOT follow the edge
            return nullptr;
        }
        if (BB == ctx.getContinueTarget()) {
            region = m_func.createContinueRegion();

            // do NOT follow the edge
            return nullptr;
        }
    }

    llvm::TerminatorInst *termInst = BB->getTerminator();

    if (llvm::ReturnInst *retInst = llvm::dyn_cast<llvm::ReturnInst>(termInst)) {
        Region *head = m_func.getTopRegion(BB);
        region = m_func.createReturnRegion(head, retInst);

        // no successor
        return nullptr;
    } else if (llvm::isa<llvm::UnreachableInst>(termInst)) {
        Region *head = m_func.getTopRegion(BB);
        region = m_func.createInvalidRegion(head);

        // no successor
        return nullptr;
    }

    // handle cycles first
    llvm::Loop *loop = m_func.getLoopFor(BB);
    if (loop != nullptr && loop != ctx.get_parentLoop()) {
        // we just entered a new loop

        bool more_then_one = false;
        BasicBlock *ExitBlock = getLoopUniqueExitTarget(*loop, more_then_one);

        MDL_ASSERT(!more_then_one && "Loop without unique exit block");

        RegionContext loopCtx(
            /*exitBlock=*/     BB,
            /*parentLoop=*/    loop,
            /*continueTarget=*/BB,
            /*breakTarget=*/   ExitBlock);

        return processLoop(loopCtx, /*head=*/BB, region);
    }

    switch (termInst->getNumSuccessors()) {
    case 1:
        region = m_func.getTopRegion(BB);
        return termInst->getSuccessor(0);

    case 2:
        // should be an if
        {
            IfContext *ifCtx = IfContext::create(ctx, m_func, BB);

            BasicBlock *exitBlock = nullptr;
            while (ifCtx->handleProperRegions(ctx, exitBlock)) {
                delete ifCtx;

                ifCtx = IfContext::create(ctx, m_func, BB);
            }

            region = ifCtx->createIfRegion(*this, exitBlock, termInst);
            delete ifCtx;

            if (exitBlock == ctx.getExitBlock())
                return nullptr;
            return exitBlock;
        }
    default:
        MDL_ASSERT(!"unsupported control flow");
        return nullptr;
    }
}

// Discover the region started at entry,
Region *RegionBuilder::discoverRegion(
    RegionContext const &ctx,
    bool                processLoopBody,
    BasicBlock          *entry)
{
    BasicBlock *BB          = entry;
    Region     *childRegion = nullptr;

    SmallVector<Region *, 4> regions;

    do {
        BasicBlock *next = processBB(ctx, processLoopBody, BB, childRegion);

        // if we was inside a loop, we left it now
        processLoopBody = false;

        regions.push_back(childRegion);
        BB = next;
    } while (BB != nullptr && BB != ctx.getExitBlock());

    if (regions.size() == 1) {
        // only one was created
        return regions[0];
    } else {
        // create a sequence
        return m_func.createSequenceRegion(
            regions[0], ArrayRef<Region *>(regions.begin() + 1, regions.end()));
    }
}

// Build regions for a function.
ASTFunction *RegionBuilder::buildRegions()
{
    RegionContext ctx;

    BasicBlock *entry = &m_func.getFunction().getEntryBlock();

    Region *body = discoverRegion(ctx, /*processLoopBody=*/false, entry);

    m_func.setBody(body);

    return &m_func;
}

// ----------------------------------------------------------------------------

// Creates a new IfContext instance.
IfContext *IfContext::create(
    RegionContext const &ctx,
    ASTFunction         &func,
    BasicBlock          *BB)
{
    BranchInst *branchInst = llvm::cast<llvm::BranchInst>(BB->getTerminator());
    BasicBlock *thenBlock = branchInst->getSuccessor(0);
    BasicBlock *elseBlock = branchInst->getSuccessor(1);

    // "then" is trivial? -> create a IF region
    if (thenBlock == ctx.getExitBlock()) {
        return new IfContext(func, BB, /*exit=*/ nullptr, ctx, nullptr, ctx, elseBlock);
    }

    // "else" is trivial? -> create a IF region
    if (elseBlock == ctx.getExitBlock()) {
        return new IfContext(func, BB, /*exit=*/ nullptr, ctx, thenBlock, ctx, nullptr);
    }

    BlockSet regularExits = ctx.getRegularExitTargets();

    BlockSet thenExits = computeDominanceFrontierExt(
        func.getDomTree(), thenBlock, regularExits);

    // join case with "else" block as an exit of the "then" region?
    if (thenExits.contains(elseBlock)) {
        // ensure, that no other exit of the "then" region is reachable from the "else" block
        bool elseReachesOtherThenExit = false;
        for (BasicBlock *thenExit : thenExits) {
            if (thenExit != elseBlock && reachable(ctx, elseBlock, thenExit)) {
                elseReachesOtherThenExit = true;
                break;
            }
        }

        // valid join case? -> create a IF region
        if (!elseReachesOtherThenExit) {
            RegionContext thenContext(elseBlock, ctx);
            return new IfContext(
                func, BB, /*exit=*/ elseBlock, thenContext, thenBlock, ctx, nullptr);
        }
    }

    BlockSet elseExits = computeDominanceFrontierExt(
        func.getDomTree(), elseBlock, regularExits);

    // join case with "then" block as an exit of the "else" region?
    if (elseExits.contains(thenBlock)) {
        // ensure, that no other exit of the "else" region is reachable from the "then" block
        bool thenReachesOtherElseExit = false;
        for (BasicBlock *elseExit : elseExits) {
            if (elseExit != thenBlock && reachable(ctx, thenBlock, elseExit)) {
                thenReachesOtherElseExit = true;
                break;
            }
        }

        // valid join case? -> create a IF region
        if (!thenReachesOtherElseExit) {
            RegionContext elseContext(thenBlock, ctx);
            return new IfContext(
                func, BB, /*exitBlock=*/ thenBlock, ctx, nullptr, elseContext, elseBlock);
        }
    }

    // general case, create a IF-ELSE region
    RegionContext thenContext(/*exitBlock=*/ nullptr, ctx);
    RegionContext elseContext(/*exitBlock=*/ nullptr, ctx);

    return new IfContext(
        func, BB, /*exit=*/ nullptr, thenContext, thenBlock, elseContext, elseBlock);
}

// Compute the exit sets of the then/else branches.
void IfContext::computeExitSets(
    BlockSet       &thenExitSet,
    BlockSet       &elseExitSet,
    BlockSet       &usedRegularExits,
    BlockSet const &regularExits)
{
    thenExitSet.clear();
    elseExitSet.clear();
    usedRegularExits.clear();

    if (m_thenBlock != nullptr) {
        thenExitSet = computeDominanceFrontierExt(
            m_func.getDomTree(), m_thenBlock, regularExits, &usedRegularExits);
    }
    if (m_elseBlock != nullptr) {
        elseExitSet = computeDominanceFrontierExt(
            m_func.getDomTree(), m_elseBlock, regularExits, &usedRegularExits);
    }
}

// Compute the set of all blocks that are reachable (dominated) by the given header until
// we reach one of the exits.
BlockSet IfContext::computeDominatedRegion(
    llvm::BasicBlock *header,
    BlockSet const   &exits)
{
    BlockSet blocks;
    BlockSet visited;

    blocks.insert(header);

    do {
        BasicBlock *block = blocks.pop();

        if (!exits.contains(block)) {
            // not an exit from this region
            if (visited.insert(block)) {
                // and not yet visited, go down
                blocks.insert(succ_begin(block), succ_end(block));
            }
        }
    } while (!blocks.empty());
    return visited;
}

// Convert an integer value.
ConstantInt *IfContext::get_constant(int i)
{
    return ConstantInt::get(m_func.get_type_mapper().get_int_type(), i);
}

/// Replaces oldBlock with newBlock in all PHI nodes in blockToFix.
static void fixPHIsInBlock(
    BasicBlock *blockToFix,
    BasicBlock *oldBlock,
    BasicBlock *newBlock)
{
    for (PHINode &phi : blockToFix->phis()) {
        int blockIdx = phi.getBasicBlockIndex(oldBlock);
        MDL_ASSERT(blockIdx > -1 && "PHI node not set to fused block!");
        phi.setIncomingBlock(blockIdx, newBlock);
    }
}

// Check if the current if-region has a single exit block, if not, try to insert one.
bool IfContext::handleProperRegions(
    RegionContext const &ctx,
    BasicBlock          *&oExitBlock)
{
    // determine the combined exit set of the child regions of the if
    BlockSet thenExitSet, elseExitSet, usedAnticipatedExits;
    computeExitSets(
        thenExitSet, elseExitSet, usedAnticipatedExits, ctx.getAnticipatedExitTargets());

    BlockSet exitSet;
    exitSet.insert(thenExitSet.begin(), thenExitSet.end());
    exitSet.insert(elseExitSet.begin(), elseExitSet.end());

    // add the context exit block to the exit set (if any), if it was used as a boundary during
    // the exit set calculation and it's neither the break nor the continue target
    // TODO: why break and continue target is excluded?
    if (ctx.getExitBlock() != nullptr &&
            ctx.getExitBlock() != ctx.getBreakTarget() &&
            ctx.getExitBlock() != ctx.getContinueTarget() &&
            usedAnticipatedExits.contains(ctx.getExitBlock())) {
        exitSet.insert(ctx.getExitBlock());
    }

    if (exitSet.size() == 0) {
        oExitBlock = ctx.getExitBlock();
    } else if (exitSet.size() == 1) {
        oExitBlock = exitSet.front();
    } else {
        // single exit node condition is violated.
        // fix it by letting all blocks in target region jumping to exit nodes jump to a new "fused"
        // block instead.
        // the fused block then jumps to the original exit nodes.
        BlockSet targetRegion = computeDominatedRegion(m_entry, exitSet);

        Function *func = &m_func.getFunction();
        LLVMContext &llvmCtx = func->getContext();
        BasicBlock *fusedBlock = BasicBlock::Create(llvmCtx, "fused", func);
        m_func.createBasicRegion(fusedBlock);
        PHINode *indexPhi = PHINode::Create(
            m_func.get_type_mapper().get_int_type(), 0, "index", fusedBlock);
        int curIndex = 0;

        SmallVector<int, 8> phiIndicesToRemove;

        for (BasicBlock *curExitBlock : exitSet) {
            Value *curIndexVal = get_constant(curIndex);

            std::map<BasicBlock *, BasicBlock *> predToIntermediateMap;
            // note: iterate carefully, as we change the block use list by changing the terminator
            for (pred_iterator predIt = pred_begin(curExitBlock);
                    predIt != pred_end(curExitBlock); ) {
                BasicBlock *exitPred = *predIt;
                ++predIt;

                // not part of target region? -> predecessor does not need to be updated
                if (!targetRegion.contains(exitPred))
                    continue;

                // check whether we already jumped from this predecessor to the fused block
                // for another exit block
                int exitPredPhiIndex = indexPhi->getBasicBlockIndex(exitPred);
                if (exitPredPhiIndex != -1 &&
                        indexPhi->getIncomingValue(exitPredPhiIndex) != curIndexVal) {
                    // -> critical edge, we need to create an intermediate jump target
                    BasicBlock *intermediateBlock =
                        BasicBlock::Create(llvmCtx, "intermediate", func);
                    m_func.createBasicRegion(intermediateBlock);
                    BranchInst::Create(fusedBlock, intermediateBlock);
                    predToIntermediateMap[exitPred] = intermediateBlock;

                    // mark where to jump to from the fused block if coming from the
                    // intermediate block
                    indexPhi->addIncoming(curIndexVal, intermediateBlock);

                    // let this predecessor jump to the intermediate block instead of
                    // the current exit block
                    TerminatorInst *termInst = exitPred->getTerminator();
                    for (unsigned i = 0, n = termInst->getNumSuccessors(); i < n; ++i) {
                        if (termInst->getSuccessor(i) == curExitBlock) {
                            termInst->setSuccessor(i, intermediateBlock);
                        }
                    }
                } else {
                    // mark where to jump to from the fused block if coming from this predecessor
                    indexPhi->addIncoming(curIndexVal, exitPred);

                    // let this predecessor jump to the fused block instead of current exit block
                    TerminatorInst *termInst = exitPred->getTerminator();
                    for (unsigned i = 0, n = termInst->getNumSuccessors(); i < n; ++i) {
                        if (termInst->getSuccessor(i) == curExitBlock) {
                            termInst->setSuccessor(i, fusedBlock);
                        }
                    }
                }
            }

            // create phis in fused block for all phis in the current exit block
            // note: iterate carefully, as we may remove PHI nodes
            for (BasicBlock::iterator bbIt = curExitBlock->begin(); bbIt != curExitBlock->end(); ) {
                Instruction *inst = &*bbIt;
                PHINode *phi = llvm::dyn_cast<PHINode>(inst);
                if (phi == nullptr)
                    break;  // end of phi list at beginning of block
                ++bbIt;

                PHINode *newPhi = PHINode::Create(phi->getType(), 0, "newPhi", fusedBlock);
                phi->addIncoming(newPhi, fusedBlock);

                phiIndicesToRemove.clear();
                for (unsigned i = 0, n = phi->getNumOperands(); i < n; ++i) {
                    BasicBlock *incomingBlock = phi->getIncomingBlock(i);

                    // was this incoming block not handled? -> skip it
                    if (!targetRegion.contains(incomingBlock))
                        continue;

                    // check whether we created an intermediate block for this predecessor
                    std::map<BasicBlock *, BasicBlock *>::const_iterator it =
                        predToIntermediateMap.find(incomingBlock);
                    if (it != predToIntermediateMap.end())
                        incomingBlock = it->second;

                    newPhi->addIncoming(phi->getIncomingValue(i), incomingBlock);
                    phiIndicesToRemove.push_back(i);
                }

                // iterate from back to front to ensure the indices stay valid
                for (int index : reverse(phiIndicesToRemove)) {
                    phi->removeIncomingValue(index, /*DeletePHIIfEmpty=*/ true);
                }

                // replace trivial phis by their incoming value
                if (phi->getNumOperands() == 1) {
                    phi->replaceAllUsesWith(phi->getIncomingValue(0));
                    phi->eraseFromParent();
                }
            }

            ++curIndex;
        }

        // all exit blocks handled, fill up phis in the fused block with undefs for unhandled
        // incoming blocks
        for (Instruction &inst : *fusedBlock) {
            PHINode *phi = llvm::dyn_cast<PHINode>(&inst);
            if (phi == nullptr)
                break;  // end of phi list at beginning of block

            Value *undef = UndefValue::get(phi->getType());
            for (BasicBlock *pred : predecessors(fusedBlock)) {
                int predIdx = phi->getBasicBlockIndex(pred);

                // no value for this predecessor, yet? -> add undef for it
                if (predIdx == -1) {
                    phi->addIncoming(undef, pred);
                }
            }
        }

        // create an if cascade for the target blocks of the fused block
        BasicBlock *curSrcBlock = fusedBlock;
        curIndex = 0;

        for (BasicBlock *curExitBlock : exitSet) {
            Value *curIndexVal = get_constant(curIndex);
            Value *cmp = CmpInst::Create(
                Instruction::ICmp,
                ICmpInst::ICMP_EQ,
                indexPhi,
                curIndexVal,
                "indexCmp",
                curSrcBlock);

            if (curSrcBlock != fusedBlock) {
                fixPHIsInBlock(curExitBlock, fusedBlock, curSrcBlock);
            }

            // pre-last block? -> end of cascade
            if (curIndex == exitSet.size() - 2) {
                BasicBlock *lastExitBlock = exitSet.back();

                if (lastExitBlock != fusedBlock)
                    fixPHIsInBlock(lastExitBlock, fusedBlock, curSrcBlock);

                BranchInst::Create(curExitBlock, lastExitBlock, cmp, curSrcBlock);
                break;
            }

            // create next block of cascade and jump there in false-case
            BasicBlock *next = BasicBlock::Create(llvmCtx, "fusedCascade", func);
            m_func.createBasicRegion(next);
            BranchInst::Create(curExitBlock, next, cmp, curSrcBlock);
            curSrcBlock = next;

            ++curIndex;
        }

        oExitBlock = fusedBlock;

        // we changed the CFG, so update the dominance tree
        m_func.updateDomTree();
    }

    // set the exit block, in the available child regions
    if (m_thenBlock != nullptr)
        m_thenContext.setExitBlock(oExitBlock);
    if (m_elseBlock != nullptr)
        m_elseContext.setExitBlock(oExitBlock);

    // return whether we ran into the fused-block case
    return exitSet.size() > 1;
}

Region *IfContext::createIfRegion(
    RegionBuilder  &builder,
    BasicBlock     *exitBlock,
    TerminatorInst *terminator)
{
    Region *thenRegion = nullptr;
    if (m_thenBlock != nullptr) {
        thenRegion = builder.discoverRegion(
            m_thenContext,
            /*processLoopBody=*/false,
            m_thenBlock);
    }

    Region *elseRegion = nullptr;
    if (m_elseBlock != nullptr) {
        elseRegion = builder.discoverRegion(
            m_elseContext,
            /*processLoopBody=*/false,
            m_elseBlock);
    }

    ASTFunction &func = builder.m_func;
    Region      *head = func.getTopRegion(m_entry);

    if (thenRegion == nullptr || elseRegion == nullptr) {
        return func.createIfThenRegion(
            head,
            thenRegion != nullptr ? thenRegion : elseRegion,
            terminator,
            /*negated=*/ thenRegion == nullptr);
    } else {
        return func.createIfThenElseRegion(head, thenRegion, elseRegion, terminator);
    }
}

// ----------------------------------------------------------------------------

// Constructor.
ASTComputePass::ASTComputePass(mi::mdl::Type_mapper &type_mapper)
: ModulePass(ID)
, m_type_mapper(type_mapper)
{
}

// Destructor.
ASTComputePass::~ASTComputePass()
{
    for (auto it : m_ast_function_map) {
        delete it.second;
    }
}

void ASTComputePass::getAnalysisUsage(AnalysisUsage &usage) const
{
    usage.addRequired<DominatorTreeWrapperPass>();
    usage.addRequired<LoopInfoWrapperPass>();
    usage.setPreservesAll();
}

bool ASTComputePass::runOnModule(Module &M)
{
    for (Function &func : M.functions()) {
        if (func.isDeclaration())
            continue;

        RegionBuilder RB(
            func,
            m_type_mapper,
            getAnalysis<DominatorTreeWrapperPass>(func).getDomTree(),
            getAnalysis<LoopInfoWrapperPass>(func).getLoopInfo());

        m_ast_function_map[&func] = RB.buildRegions();
    }
    return false;
}

char ASTComputePass::ID = 0;

//------------------------------------------------------------------------------
Pass *createASTComputePass(mi::mdl::Type_mapper &type_mapper)
{
    return new ASTComputePass(type_mapper);
}

unsigned RegionBuilder::dumpRegion(FILE *file, Region const *region)
{
    std::string kind_str = getKindString(region);
    fprintf(file, "n%u [label=\"%s %u %s\"];\n",
        unsigned(region->get_id()), kind_str.c_str(), unsigned(region->get_id()),
        std::string(region->get_bb()->getName()).c_str());
    return unsigned(region->get_id());
}

// Dump an edge.
void dumpEdge(
    FILE *file,
    ArrayRef<unsigned> const &sources,
    unsigned target,
    char const *label=nullptr)
{
    for (unsigned source : sources) {
        fprintf(file, "n%u -> n%u", source, target);
        if (label)
            fprintf(file, " [label=\"%s\";]", label);
        fprintf(file, ";\n");
    }
}

// Dump an edge.
void dumpEdgeToRegion(
    FILE *file,
    ArrayRef<unsigned> const &sources,
    Region *target_region,
    char const *label=nullptr)
{
    dumpEdge(file, sources, unsigned(target_region->getEntryRegionBlock()->get_id()), label);
}

// Dump a sub graph.
std::vector<unsigned> RegionBuilder::dumpSubGraph(FILE *file, Region const *region)
{
    typedef std::vector<unsigned> ExitSet;
    ExitSet res;

    if (region->get_kind() == Region::SK_BLOCK) {
        res.push_back(dumpRegion(file, region));
        return res;
    }

    std::string kind_str = getKindString(region);
    Region const *top_region = region->getEntryRegionBlock();
    fprintf(file, "subgraph cluster_%u { label=\"%s %u %s\";\n",
        unsigned(region->get_id()), kind_str.c_str(), unsigned(region->get_id()),
        std::string(top_region->get_bb()->getName()).c_str());

    switch (region->get_kind()) {
    case Region::SK_BREAK:
        break;

    case Region::SK_CONTINUE:
        break;

    case Region::SK_RETURN:
        {
            RegionReturn const *r = static_cast<RegionReturn const *>(region);
            res = dumpSubGraph(file, r->getHead());
        }
        break;

    case Region::SK_SEQUENCE:
        {
            RegionSequence const *r = static_cast<RegionSequence const *>(region);
            ExitSet a = dumpSubGraph(file, r->getHead());
            for (Region *next : r->getTail()) {
                ExitSet b = dumpSubGraph(file, next);

                dumpEdgeToRegion(file, a, next);
                a = b;
            }
            res = a;
        }
        break;

    case Region::SK_IF_THEN:
        {
            RegionIfThen const *r = static_cast<RegionIfThen const *>(region);
            ExitSet a = dumpSubGraph(file, r->getHead());
            ExitSet b = dumpSubGraph(file, r->getThen());
            dumpEdgeToRegion(file, a, r->getThen(), "then");
            res = a;
            if (!r->getThen()->is_jump())
                res.insert(res.end(), b.begin(), b.end());
        }
        break;

    case Region::SK_IF_THEN_ELSE:
        {
            RegionIfThenElse const *r = static_cast<RegionIfThenElse const *>(region);
            ExitSet a = dumpSubGraph(file, r->getHead());
            ExitSet b = dumpSubGraph(file, r->getThen());
            ExitSet c = dumpSubGraph(file, r->getElse());
            dumpEdgeToRegion(file, a, r->getThen(), "then");
            dumpEdgeToRegion(file, a, r->getElse(), "else");
            res = b;
            res.insert(res.end(), c.begin(), c.end());
        }
        break;

    case Region::SK_NATURAL_LOOP:
        {
            RegionNaturalLoop const *r = static_cast<RegionNaturalLoop const *>(region);
            ExitSet a = dumpSubGraph(file, r->getHead());
            dumpEdgeToRegion(file, a, r->getHead(), "backedge");
            res = a;
        }
        break;

    default:
        break;
    }

    fprintf(file, "}\n");

    return res;
}

// Dump the current graph.
void RegionBuilder::dumpRegionGraph(char const *suffix, bool with_subgraphs)
{
#ifdef DUMP_REGIONGRAPHS
    char buf[16];
    snprintf(buf, sizeof(buf), "%.3u", unsigned(m_dump_id));
    std::string filename = m_dump_base_name + "-" + buf;
    if (suffix) filename += std::string("-") + suffix;
    filename += ".gv";
    FILE *file = fopen(filename.c_str(), "wt");

    fprintf(file, "digraph \"%s-%u\" {\n", m_dump_base_name.c_str(), unsigned(m_dump_id));

    for (Region const *region : m_work_list) {
#ifdef DUMP_STMTKINDS
        std::string kind_str = getKindString(region);

        std::string head_block_name;

        if (with_subgraphs && region->get_kind() != Region::SK_BLOCK) {
            dumpSubGraph(file, region);
        } else {
            Region const *top_region = region->getEntryRegionBlock();
            fprintf(file, "n%u [label=\"%s %u %s",
                unsigned(region->get_id()), kind_str.c_str(), unsigned(region->get_id()),
                std::string(top_region->get_bb()->getName()).c_str());

            if (size_t copied_from = region->copied_from())
                fprintf(file, " (%u)", unsigned(copied_from));
            fprintf(file, "\"");
            if (region->is_loop_head())
                fprintf(file, " shape=box");
            else if (region->is_loop())
                fprintf(file, " shape=diamond");
            fprintf(file, " ]\n");
        }
#else
#ifdef DUMP_BLOCKNAMES
        fprintf(file, "n%u [label=\"%s",
            unsigned(region->get_id()), std::string(region->blocks()[0]->getName()).c_str());
#else
        fprintf(file, "n%u [label=\"%u",
            unsigned(region->get_id()), unsigned(region->get_id()));
#endif
#endif
        /*if (size_t copied_from = region->copied_from())
            fprintf(file, " (%u)", unsigned(copied_from));
        fprintf(file, "\"");
        if (region->is_loop_head())
            fprintf(file, " shape=box");
        else if (region->is_loop())
            fprintf(file, " shape=diamond");
        fprintf(file, " ]\n");*/
    }

    for (Region const *region : m_work_list) {
        unsigned node_id = unsigned(region->get_id());

        for (Region const *succ : region->succs()) {
            fprintf(file, "n%u -> n%u\n", node_id, unsigned(succ->get_id()));

            fprintf(file, "\n");
        }
    }

    fprintf(file, "}\n");

    fclose(file);

    ++m_dump_id;
#endif
}


LoopExitEnumerationPass::LoopExitEnumerationPass()
    : FunctionPass( ID )
{
}

void LoopExitEnumerationPass::getAnalysisUsage(AnalysisUsage &usage) const
{
    usage.addRequired<LoopInfoWrapperPass>();
}

/// Based on a reference PHI, fill up all PHI nodes in the given block with undef values
/// for any missing predecessors.
static void fillUpPhis(BasicBlock *bb, PHINode *reference_phi)
{
    // collect all known predecessor blocks
    std::map<BasicBlock *, unsigned> visited_map;
    for (BasicBlock *pred : reference_phi->blocks()) {
        visited_map[pred] = 0;
    }

    unsigned visit_id = 1;
    for (BasicBlock::iterator new_exit_it = bb->begin(); isa<PHINode>(new_exit_it); ++new_exit_it) {
        // skip the pred_id PHI node, which is full per definition
        if (&*new_exit_it == reference_phi)
            continue;

        PHINode *phi = cast<PHINode>(new_exit_it);

        // mark all already handled blocks
        for (BasicBlock *block : phi->blocks()) {
            visited_map[block] = visit_id;
        }

        // add all missing blocks
        for (auto pair : visited_map) {
            // already handled? -> skip
            if (pair.second == visit_id)
                continue;

            phi->addIncoming(UndefValue::get(phi->getType()), pair.first);
        }

        ++visit_id;
    }
}

/// Relocate all PHI nodes in src_bb to tgt_bb.
static void relocatePhis(BasicBlock *src_bb, BasicBlock *tgt_bb)
{
    for (BasicBlock::iterator exit_it = src_bb->begin(); isa<PHINode>(exit_it); ) {
        PHINode *phi = cast<PHINode>(exit_it);
        PHINode *new_phi = PHINode::Create(phi->getType(), 0, phi->getName(), tgt_bb);
        for (unsigned i = 0, n = phi->getNumIncomingValues(); i < n; ++i) {
            BasicBlock *block = phi->getIncomingBlock(i);
            Value *val = phi->getIncomingValue(i);
            new_phi->addIncoming(val, block);
        }
        phi->replaceAllUsesWith(new_phi);
        exit_it = phi->getParent()->getInstList().erase(exit_it);
    }
}

bool LoopExitEnumerationPass::runOnFunction(Function &function)
{
    bool changed = false;
    LoopInfo &loop_info = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    LLVMContext &llvm_context = function.getContext();

    // ensure that every loop only has one exit and this exit is part of the parent loop,
    // if there is a parent loop
    for (Loop *loop : loop_info.getLoopsInPreorder()) {
        SmallVector<BasicBlock *, 8> exit_blocks;
        loop->getUniqueExitBlocks(exit_blocks);

        if (exit_blocks.size() > 1) {
            // Insert an intermediate block where all exiting blocks jump to.
            // The original blocks will then be reached by a switch instruction over a predecessor
            // ID determined by the PHI node "pred_id_phi".
            // The intermediate block will still be dominated by the loop header

            IntegerType *int_type = IntegerType::get(llvm_context, 32);
            BasicBlock *new_exit_block = BasicBlock::Create(
                llvm_context, "new_exit_block", &function);
            PHINode *pred_id_phi = PHINode::Create(int_type, 0, "pred_id", new_exit_block);
            changed = true;

            SmallVector<std::pair<ConstantInt *, BasicBlock *>, 8> switch_cases;
            int exit_pred_id = 0;

            for (BasicBlock *exit_block : exit_blocks) {
                // due to the LoopSimplifyCFG pass, all predecessors are inside the loop.

                // collect the terminator instructions leading to the exit_block,
                // as we will change the users list
                SmallVector<TerminatorInst *, 8> pred_terms;
                for (auto exit_user : exit_block->users()) {
                    TerminatorInst *term = cast<TerminatorInst>(exit_user);
                    pred_terms.push_back(term);
                }

                // redirect all jumps to exit_block through our switch block
                for (TerminatorInst *term : pred_terms) {
                    BasicBlock *pred_block = term->getParent();

                    bool exit_seen = false;
                    for (int i = 0, n = term->getNumSuccessors(); i < n; ++i) {
                        // target is not the current exit_block -> skip
                        if (term->getSuccessor(i) != exit_block)
                            continue;

                        term->setSuccessor(i, new_exit_block);

                        // only add as new incoming block, if we haven't done so already
                        if (!exit_seen) {
                            exit_seen = true;

                            ConstantInt *case_id = ConstantInt::get(int_type, exit_pred_id);
                            pred_id_phi->addIncoming(case_id, pred_block);
                            switch_cases.push_back(std::make_pair(case_id, exit_block));
                        }
                    }

                    ++exit_pred_id;
                }

                // move all PHI nodes to our new block
                relocatePhis(exit_block, new_exit_block);
            }

            // create switch and use first case as default case
            SwitchInst *new_switch = SwitchInst::Create(
                pred_id_phi, switch_cases[0].second, exit_blocks.size(), new_exit_block);
            for (size_t i = 1, n = switch_cases.size(); i < n; ++i) {
                auto &switch_case = switch_cases[i];
                new_switch->addCase(switch_case.first, switch_case.second);
            }

            // ensure, that all PHI nodes have the right number of operands, filling up with undef
            fillUpPhis(new_exit_block, pred_id_phi);
        }
    }

    return changed;
}

char LoopExitEnumerationPass::ID = 0;

Pass* createLoopExitEnumerationPass()
{
    return new LoopExitEnumerationPass();
}



UnswitchPass::UnswitchPass()
: FunctionPass( ID )
{
}

void UnswitchPass::getAnalysisUsage(AnalysisUsage &usage) const
{
}

// Fixes the PHI nodes in the given block, when the predecessor old_pred is replaced by new_pred.
void UnswitchPass::fixPhis(BasicBlock *bb, BasicBlock *old_pred, BasicBlock *new_pred)
{
    for (PHINode &phi : bb->phis()) {
        int idx = phi.getBasicBlockIndex(old_pred);
        while (idx != -1) {
            phi.setIncomingBlock(unsigned(idx), new_pred);
            idx = phi.getBasicBlockIndex(old_pred);
        }
    }
}

bool UnswitchPass::runOnFunction(Function &function)
{
    bool changed = false;
    LLVMContext &llvm_context = function.getContext();

    // always insert new blocks after "end" to avoid iterating over them
    for (Function::iterator it = function.begin(), end = function.end(); it != end; ++it) {
        if (SwitchInst *switch_inst = dyn_cast<SwitchInst>(it->getTerminator())) {
            BasicBlock *cur_bb = &*it;
            Value *switch_cond = switch_inst->getCondition();
            for (auto cur_case : switch_inst->cases()) {
                BasicBlock *next_bb = BasicBlock::Create(llvm_context, "unswitch_case", &function);
                Value *case_cond = CmpInst::Create(Instruction::ICmp, ICmpInst::ICMP_EQ,
                    switch_cond, cur_case.getCaseValue(), "", cur_bb);
                BranchInst::Create(cur_case.getCaseSuccessor(), next_bb, case_cond, cur_bb);
                // did predecessor for the case successor change? -> fix PHIs
                if (cur_bb != &*it) {
                    fixPhis(cur_case.getCaseSuccessor(), &*it, cur_bb);
                }
                cur_bb = next_bb;
            }

            // otherwise, unconditionally jump to default case
            BranchInst::Create(switch_inst->getDefaultDest(), cur_bb);
            fixPhis(switch_inst->getDefaultDest(), &*it, cur_bb);

            changed = true;
            switch_inst->eraseFromParent();
        }
    }

    return changed;
}

char UnswitchPass::ID = 0;

Pass *createUnswitchPass()
{
    return new UnswitchPass();
}

}  // namespace hlsl
}  // namespace llvm
