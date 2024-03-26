/******************************************************************************
 * Copyright (c) 2018-2024, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_GENERATOR_JIT_SL_FUNCTION_H
#define MDL_GENERATOR_JIT_SL_FUNCTION_H 1

#include <vector>
#include <map>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/ilist_node.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/Function.h>

namespace mi {
namespace mdl {
// forward
class Type_mapper;
}
}

namespace llvm {

// forward
class BasicBlock;
class Loop;
class LoopInfo;
class DominatorTree;

namespace sl {

// forward
class StructuredFunction;
class Region;
class RegionComplex;

// Base class for all regions in the Structured Function.
class Region : public ilist_node_with_parent<Region, StructuredFunction> {
    friend class RegionBuilder;
    friend class StructuredFunction;

public:
    typedef std::vector<Region *>     RegionList;
    typedef std::vector<BasicBlock *> BB_list;

    /// The Kinds of a region.
    enum Kind {
        SK_INVALID,          ///< Invalid region, used for sentinels only.
        SK_BLOCK,            ///< the smallest unit, one (basic) block
        SK_SEQUENCE,         ///< a sequence of regions
        SK_IF_THEN,          ///< an if-then region
        SK_IF_THEN_ELSE,     ///< an if-then-else region
        SK_NATURAL_LOOP,     ///< a natural loop
        SK_BREAK,            ///< a region ending with a break jump
        SK_CONTINUE,         ///< a region ending with a continue jump
        SK_RETURN,           ///< a region ending with a return
        SK_SWITCH,           ///< a switch region
    };

    /// Flags for SCC.
    enum Flags {
        ON_STACK  = 1 << 0,
        ON_LOOP   = 1 << 1,
        IS_HEADER = 1 << 2,
    };

protected:
    /// Default constructor.
    Region()
    : m_parent(nullptr)
    , m_owner(nullptr)
    , m_id(0)
    , m_copied_from(0)
    , m_visit_count(0)
    , m_bb(nullptr)
    , m_dfs_num(0)
    , m_low(0)
    , m_flags(0)
    {
    }

    /// Constructor from a basic block.
    ///
    /// \param parent  the function owner of this region
    /// \param id      the unique ID of this region
    /// \param bb      the basic block
    Region(
        StructuredFunction *parent,
        size_t             id,
        BasicBlock         *bb)
    : m_parent(parent)
    , m_owner(nullptr)
    , m_id(id)
    , m_copied_from(0)
    , m_visit_count(0)
    , m_bb(bb)
    , m_dfs_num(0)
    , m_low(0)
    , m_flags(0)
    {
    }

    /// Destructor.
    virtual ~Region() {}

public:
    /// Get the kind of the region.
    virtual Kind get_kind() const = 0;

    /// Get the parent.
    /// Note: Must be named getParent(), otherwise the LLVM dominance builder will not work.
    StructuredFunction *getParent() { return m_parent; }

    /// Set the parent.
    void setParent(StructuredFunction *parent) { m_parent = parent; }

    /// Get the owning region, if any.
    RegionComplex *getOwnerRegion() { return m_owner; }

    /// Set the owning region.
    void setOwnerRegion(RegionComplex *owner) { m_owner = owner; }

    /// Get the entry region of this region.
    Region const *getEntryRegionBlock() const;

    /// Get the ID of this node.
    size_t get_id() const { return m_id; }

    /// Get the ID of the "original" node this node was copied from (if != zero).
    size_t copied_from() const { return m_copied_from; }

    /// Add a predecessor edge to node pred.
    void add_pred(Region *pred) {
        // allows self loops AND multiple edges
        m_preds.push_back(pred);
    }

    /// Adds all predecessors of the given region as predecessors of this region.
    ///
    /// \param node  the region we copy the predecessors from
    void add_all_preds(Region *node) {
        for (Region *pred : node->preds()) {
            add_pred(pred);
        }
    }

    /// Get all predecessors.
    RegionList &preds() { return m_preds; }

    /// Get all predecessors.
    RegionList const &preds() const { return m_preds; }

    /// Get the number of predecessors.
    size_t pred_size() const { return m_preds.size(); }

    /// Returns true if this block has at least one predecessor.
    bool has_pred() const { return !m_preds.empty(); }

    /// Get the predecessor at index i.
    Region *get_pred(size_t i) { return m_preds[i]; }

    /// Get the only predecessor, or return \c nullptr if the node does not have exactly one
    /// predecessor.
    Region *get_unique_pred() const {
        if (m_preds.size() != 1) {
            return nullptr;
        }
        return m_preds[0];
    }

    /// Erase the given node from the predecessors, if present.
    void erase_pred(Region *pred) {
        auto it = std::find(m_preds.begin(), m_preds.end(), pred);
        if (it != m_preds.end()) {
            m_preds.erase(it);
        }
    }


    /// Add a successor edge to node succ.
    void add_succ(Region *succ) {
        m_succs.push_back(succ);
    }

    /// Get all successors.
    RegionList &succs() { return m_succs; }

    /// Get all successors.
    RegionList const &succs() const { return m_succs; }

    /// Get the number of successors.
    size_t succ_size() const { return m_succs.size(); }

    /// Returns true if this block has at least one successor.
    bool has_succ() const { return !m_succs.empty(); }

    /// Get the successor at index i.
    Region *get_succ(size_t i) const { return m_succs[i]; }

    /// Get the only successor, or return \c nullptr if the node does not have exactly one successor.
    Region *get_unique_succ() const {
        if (m_succs.size() != 1) {
            return nullptr;
        }
        return m_succs[0];
    }

    /// Erase the node from the successors, if present.
    void erase_succ(Region *succ) {
        auto it = std::find(m_succs.begin(), m_succs.end(), succ);
        if (it != m_succs.end()) {
            m_succs.erase(it);
        }
    }


    /// Add all predecessors of the given node as successor and remove the node from those.
    /// Skip the given node, if found as predecessor.
    void replace_as_succ_of_preds(Region *node) {
        for (Region *pred : node->preds()) {
            if (pred == node) {
                continue;
            }
            add_pred(pred);
            pred->erase_succ(node);
            pred->add_succ(this);
        }
    }

    /// Add all predecessors of the given node as successor and remove the node from those.
    /// Skip the given node and the ignore_pred node, if found as predecessor.
    void replace_as_succ_of_preds_except(Region *node, Region *ignore_pred) {
        for (Region *pred : node->preds()) {
            if (pred == node || pred == ignore_pred) {
                continue;
            }
            add_pred(pred);
            pred->erase_succ(node);
            pred->add_succ(this);
        }
    }

    /// Add all successors of the given node as predecessor and remove the node from those.
    /// Skip the given node, if found as successor.
    void replace_as_pred_of_succs(Region *node) {
        for (Region *succ : node->succs()) {
            if (succ == node) {
                continue;
            }
            add_succ(succ);
            succ->erase_pred(node);
            succ->add_pred(this);
        }
    }

    /// Add all successors of the given node as predecessor and remove the node from those.
    /// Skip the given node and the ignore_succ node, if found as successor.
    void replace_as_pred_of_succs_except(Region *node, Region *ignore_succ) {
        for (Region *succ : node->succs()) {
            if (succ == node || succ == ignore_succ) {
                continue;
            }
            add_succ(succ);
            succ->erase_pred(node);
            succ->add_pred(this);
        }
    }

    /// Get the basic block.
    llvm::BasicBlock *get_bb() const { return m_bb; }

    /// Get the head basic block.
    virtual llvm::BasicBlock *get_head_bb() const { return get_bb(); }

    /// Get the LLVM loop information for this node.
    llvm::Loop *get_loop();

    /// Mark a node on the stack.
    void markOnStack(bool flag) { if (flag) m_flags |= ON_STACK; else m_flags &= ~ON_STACK; }

    /// Check if we are inside the stack.
    bool onStack() const { return (m_flags & ON_STACK) != 0; }

    /// mark a node inside a loop.
    void markLoop(bool flag) { if (flag) m_flags |= ON_LOOP; else m_flags &= ~ON_LOOP; }

    /// Check if we are inside a loop.
    bool is_loop() const { return (m_flags & ON_LOOP) != 0; }

    /// Check if we are inside a jump region.
    bool is_jump() const {
        Kind k = get_kind();
        return k == SK_BREAK || k == SK_CONTINUE || k == SK_RETURN;
    }

    /// Mark a node inside a loop as a header.
    void mark_loop_head(bool flag) { if (flag) m_flags |= IS_HEADER; else m_flags &= ~IS_HEADER; }

    /// Check if this node is a loop header.
    bool is_loop_head() const { return (m_flags & IS_HEADER) != 0; }

    /// Return the number of children in this node.
    virtual size_t getChildSize() const {
        return 0;
    }

    /// Get the i-th child in this node.
    virtual Region *getChild(size_t i) const {
        return nullptr;
    }

protected:
    /// The (function) parent.
    StructuredFunction   *m_parent;

    /// The owning region, if any.
    RegionComplex        *m_owner;

    /// Unique ID of this node, != zero.
    size_t                 m_id;

    /// If != zero, the ID of the original node that was copied into this one.
    size_t                 m_copied_from;

    /// Visit count for traversal.
    size_t                 m_visit_count;

    /// Predecessor edges (dependency).
    RegionList               m_preds;

    /// Successor edges (control flow).
    RegionList               m_succs;

    /// The basic block of this node.
    BasicBlock       *m_bb;

    // ---------  SCC helper --------------

    /// DFS number this node was reached first.
    size_t                 m_dfs_num;

    /// Minimum DFS number over this node and successors.
    size_t                 m_low;

    /// Node flags
    unsigned               m_flags;
};

/// Base class for all complex regions.
class RegionComplex : public Region {
    typedef Region Base;

protected:
    /// Constructor.
    RegionComplex(
        StructuredFunction *parent,
        size_t             id,
        Region             *head)
    : Base(parent, id, nullptr)
    , m_head(head)
    {
        head->setOwnerRegion(this);
    }

public:
    /// Get the head node.
    Region *getHead() const { return m_head; }

    /// Get the head basic block.
    llvm::BasicBlock *get_head_bb() const override { return getHead()->get_head_bb(); }

    /// Return the number of children in this node.
    size_t getChildSize() const override {
        return 1;
    }

    /// Get the i-th child in this node.
    Region *getChild(size_t i) const override {
        return getHead();
    }

private:
    /// The Head block.
    Region *m_head;
};

/// Base class for continue, break, or return regions.
class RegionJump : public RegionComplex {
    typedef RegionComplex Base;
    friend class StructuredFunction;

protected:
    /// Constructor.
    RegionJump(
        StructuredFunction *parent,
        size_t             id,
        Region             *head)
    : Base(parent, id, head)
    {
    }
};

/// A basic block region.
class RegionBlock : public Region {
    typedef Region Base;
    friend class StructuredFunction;

protected:
    /// Constructor from a basic block.
    RegionBlock(
        StructuredFunction *parent,
        size_t             id,
        BasicBlock         *bb)
    : Base(parent, id, bb)
    {}

public:
    /// The kind of this subclass.
    static Kind const s_kind = SK_BLOCK;

    /// Get the kind of the region.
    Kind get_kind() const final { return s_kind; }
};

/// An invalid (dead) region.
class RegionInvalid : public RegionComplex {
    typedef RegionComplex Base;
    friend class StructuredFunction;

protected:
    /// Constructor.
    RegionInvalid(
        StructuredFunction *parent,
        size_t             id,
        Region             *head)
    : Base(parent, id, head)
    {
    }

public:
    /// The kind of this subclass.
    static Kind const s_kind = SK_INVALID;

    /// Get the kind of the region.
    Kind get_kind() const final { return s_kind; }
};

/// A Sequence region.
class RegionSequence : public RegionComplex {
    typedef RegionComplex Base;
    friend class StructuredFunction;

protected:
    /// Constructor.
    RegionSequence(
        StructuredFunction       *parent,
        size_t                   id,
        Region                   *head,
        ArrayRef<Region *> const &tail)
    : Base(parent,  id, head)
    , m_tail(tail.begin(), tail.end())
    {
        for (Region *r : m_tail) {
            r->setOwnerRegion(this);
        }
    }

public:
    /// The kind of this subclass.
    static Kind const s_kind = SK_SEQUENCE;

    /// Get the kind of the region.
    Kind get_kind() const final { return s_kind; }

    /// Get the tail node.
    std::vector<Region *> const &getTail() const { return m_tail; }

    /// Returns the last region of this sequence.
    Region *getLastRegion() const {
        if (m_tail.empty()) {
            return getHead();
        }
        return m_tail[m_tail.size() - 1];
    }

    /// Return the number of children in this node.
    size_t getChildSize() const override {
        return 1 + m_tail.size();
    }

    /// Get the i-th child in this node.
    Region *getChild(size_t i) const override {
        if (i == 0) {
            return getHead();
        }
        return m_tail[i - 1];
    }

private:
    /// The tail block.
    std::vector<Region *> m_tail;
};

/// A If-Then region.
class RegionIfThen : public RegionComplex {
    typedef RegionComplex Base;
    friend class StructuredFunction;

protected:
    /// Constructor.
    RegionIfThen(
        StructuredFunction *parent,
        size_t             id,
        Region             *head,
        Region             *then,
        Instruction        *terminator,
        bool               negated)
    : Base(parent, id, head)
    , m_terminator_inst(terminator)
    , m_then(then)
    , m_negated(negated)
    {
        m_then->setOwnerRegion(this);
    }

public:
    /// The kind of this subclass.
    static Kind const s_kind = SK_IF_THEN;

    /// Get the kind of the region.
    Kind get_kind() const override { return s_kind; }

    /// Get the then node.
    Region *getThen() const { return m_then; }

    /// Return the number of children in this node.
    size_t getChildSize() const override {
        return 2;
    }

    /// Get the i-th child in this node.
    Region *getChild(size_t i) const override {
        return i == 0 ? getHead() : m_then;
    }

    /// Returns true, if the condition must be negated.
    bool isNegated() const {
        return m_negated;
    }

    /// Get the terminator instruction.
    Instruction *get_terminator_inst() const { return m_terminator_inst; }

private:
    /// The terminator instruction providing the condition for the if.
    Instruction *m_terminator_inst;

    /// The then block.
    Region *m_then;

    /// If true, the condition must be negated.
    bool m_negated;
};

/// A If-Then-Else region.
class RegionIfThenElse : public RegionIfThen {
    typedef RegionIfThen Base;
    friend class StructuredFunction;

protected:
    /// Constructor.
    RegionIfThenElse(
        StructuredFunction *parent,
        size_t             id,
        Region             *head,
        Region             *then_node,
        Region             *else_node,
        Instruction        *terminator)
    : Base(parent, id, head, then_node, terminator, false)
    , m_else(else_node)
    {
        m_else->setOwnerRegion(this);
    }

public:
    /// The kind of this subclass.
    static Kind const s_kind = SK_IF_THEN_ELSE;

    /// Get the kind of the region.
    Kind get_kind() const final { return s_kind; }

    /// Get the else region.
    Region *getElse() const { return m_else; }

    /// Return the number of children in this region.
    size_t getChildSize() const override {
        return 3;
    }

    /// Get the i-th child in this region.
    Region *getChild(size_t i) const override {
        switch (i) {
        case 0: return getHead();
        case 1: return getThen();
        default: return getElse();
        }
    }

private:
    /// The else region.
    Region *m_else;
};

/// A natural loop region.
class RegionNaturalLoop : public RegionComplex {
    typedef RegionComplex Base;
    friend class StructuredFunction;

protected:
    /// Constructor.
    RegionNaturalLoop(
        StructuredFunction *parent,
        size_t             id,
        Region             *head)
    : Base(parent, id, head)
    {
    }

public:
    /// The kind of this subclass.
    static Kind const s_kind = SK_NATURAL_LOOP;

    /// Get the kind of the region.
    Kind get_kind() const final { return s_kind; }
};

/// A break region.
class RegionBreak : public Region {
    typedef Region Base;
    friend class StructuredFunction;

protected:
    /// Constructor.
    RegionBreak(
        StructuredFunction *parent,
        size_t             id)
    : Base(parent, id, nullptr)
    {
    }

public:
    /// The kind of this subclass.
    static Kind const s_kind = SK_BREAK;

    /// Get the kind of the region.
    Kind get_kind() const final { return s_kind; }
};

/// A continue region.
class RegionContinue : public Region {
    typedef Region Base;
    friend class StructuredFunction;

protected:
    /// Constructor.
    RegionContinue(
        StructuredFunction *parent,
        size_t             id)
    : Base(parent, id, nullptr)
    {
    }

public:
    /// The kind of this subclass.
    static Kind const s_kind = SK_CONTINUE;

    /// Get the kind of the region.
    Kind get_kind() const final { return s_kind; }
};

/// A return region.
class RegionReturn : public RegionComplex {
    typedef RegionComplex Base;
    friend class StructuredFunction;

protected:
    /// Constructor.
    RegionReturn(
        StructuredFunction *parent,
        size_t             id,
        Region             *head,
        ReturnInst         *return_inst)
    : Base(parent, id, head)
    , m_return_inst(return_inst)
    {
    }

public:
    /// The kind of this subclass.
    static Kind const s_kind = SK_RETURN;

    /// Get the kind of the region.
    Kind get_kind() const final { return s_kind; }

    /// Get the return instruction.
    ReturnInst *get_return_inst() const { return m_return_inst; }

private:
    ReturnInst *m_return_inst;
};

/// A Switch region.
class RegionSwitch : public RegionComplex {
    typedef RegionComplex Base;
    friend class StructuredFunction;

public:
    /// Descriptor for one case.
    struct CaseDescriptor {
        Region                  *case_region;
        llvm::ConstantInt const *case_value;
        bool                    fall_through;
    };
protected:
    /// Constructor.
    RegionSwitch(
        StructuredFunction             *parent,
        size_t                         id,
        Region                         *head,
        ArrayRef<CaseDescriptor> const &cases)
    : Base(parent, id, head)
    , m_cases(cases.begin(), cases.end())
    , m_has_default_case(false)
    {
        for (CaseDescriptor const &desc : cases) {
            if (desc.case_value == nullptr) {
                m_has_default_case = true;
                break;
            }
        }
    }

public:
    /// The kind of this subclass.
    static Kind const s_kind = SK_SWITCH;

    /// Get the kind of the region.
    Kind get_kind() const final { return s_kind; }

    /// Get the number of cases.
    size_t getCasesCount() const { return m_cases.size(); }

    /// Get the i'th case descriptor.
    CaseDescriptor const &getCase(size_t i) const { return m_cases[i]; }

private:
    /// The case descriptors.
    std::vector<CaseDescriptor> m_cases;

    /// True if has a default case.
    bool m_has_default_case;
};

/// Cast to subtype or return NULL if types do not match.
template<typename T>
inline T *as(Region *region) {
    return region->get_kind() == T::s_kind ? static_cast<T *>(region) : NULL;
}

/// Cast to subtype or return NULL if types do not match.
template<>
inline RegionComplex *as<RegionComplex>(Region *region) {
    switch (region->get_kind()) {
    case Region::SK_INVALID:
    case Region::SK_SEQUENCE:
    case Region::SK_IF_THEN:
    case Region::SK_IF_THEN_ELSE:
    case Region::SK_NATURAL_LOOP:
    case Region::SK_RETURN:
    case Region::SK_SWITCH:
        return static_cast<RegionComplex *>(region);
    case Region::SK_BLOCK:
    case Region::SK_BREAK:
    case Region::SK_CONTINUE:
        return nullptr;
    }
    MDL_ASSERT(!"unexpected Region kind");
    return nullptr;
}

/// Cast to subtype or return NULL if types do not match.
template<typename T>
inline T const *as(Region const *region) {
    return region->get_kind() == T::s_kind ? static_cast<T const *>(region) : NULL;
}

/// Cast to subtype or return NULL if types do not match.
template<>
inline RegionComplex const *as<RegionComplex>(Region const *region) {
    return as<RegionComplex>(const_cast<Region *>(region));
}

/// Check if a statement is of a certain type.
template<typename T>
inline bool is(Region const *region) {
    return as<T>(region) != NULL;
}

/// A static_cast with check in debug mode.
template <typename T>
inline T *cast(Region *arg) {
    MDL_ASSERT(arg == NULL || is<T>(arg));
    return static_cast<T *>(arg);
}

/// A static_cast with check in debug mode.
template <typename T>
inline T const *cast(Region const *arg) {
    MDL_ASSERT(arg == NULL || is<T>(arg));
    return static_cast<T const *>(arg);
}

/// A function with an CFG AST on top.
class StructuredFunction {
public:
    typedef std::list<Region *> RegionList;

public:
    /// Constructor.
    ///
    /// \param func         the LLVM function to wrap
    /// \param type_mapper  the type mapper of the backend
    /// \param domTree      the dominator tree for \c func
    /// \param loop_info    the loop info for \func
    StructuredFunction(
        Function              &func,
        mi::mdl::Type_mapper  &type_mapper,
        DominatorTree         &domTree,
        LoopInfo              &loop_info);

    /// Destructor.
    ~StructuredFunction();

    RegionList::iterator       begin()       { return m_region_list.begin(); }
    RegionList::const_iterator begin() const { return m_region_list.begin(); }

    RegionList::iterator       end()       { return m_region_list.end(); }
    RegionList::const_iterator end() const { return m_region_list.end(); }

    Region       *front()       { return *m_region_list.begin(); }
    Region const *front() const { return *m_region_list.begin(); }

    size_t size() const { return m_region_list.size(); }

    /// Get the associated LLVM function.
    Function &getFunction() const {
        return m_func;
    }

    /// Get the LLVM loop information of the associated LLVM function.
    LoopInfo &getLoopInfo() const {
        return m_loop_info;
    }

    /// Get the LLVM loop for a given basic block.
    ///
    /// \param bb  the basic block
    Loop *getLoopFor(
        BasicBlock const *BB) const;

    /// Get the LLVM dominator tree of the function.
    DominatorTree &getDomTree() const {
        return m_domTree;
    }

    /// Returns true iff a basic block A dominates a basic block B.
    ///
    /// \param A  basic block A
    /// \param B  basic block B
    bool dominates(
        BasicBlock *A,
        BasicBlock *B) const;

    /// Get the name of this function.
    StringRef getName() {
        return m_func.getName();
    }

    /// Returns the body of the function represented by the root region.
    Region *getBody() const {
        return m_body;
    }

    /// Set the body of the function represented by the root region.
    ///
    /// \param body  the root region of the function
    void setBody(Region *body) {
        m_body = body;
    }

    /// Get top-most region for a basic block.
    Region *getTopRegion(BasicBlock *bb);

    /// Create a new Basic region in the graph.
    ///
    /// \param bb  the basic block
    Region *createBasicRegion(BasicBlock *bb);

    /// Create a new Invalid region in the graph.
    ///
    /// \param head  the entry region
    Region *createInvalidRegion(Region *head);

    /// Create a new Sequence region in the graph.
    ///
    /// \param head   the entry region of this sequence
    /// \param tail   an ordered list of the sequence tail(s)
    RegionSequence *createSequenceRegion(
        Region                   *head,
        ArrayRef<Region *> const &tail);

    /// Create a new IfThen region in the graph.
    ///
    /// \param head        the entry region
    /// \param then        the then region
    /// \param terminator  the llvm basic block terminator instruction
    /// \param negated     if true, the jump condition in the llvm instruct must be negated to
    ///                    have a non-empty then and an empty else
    RegionIfThen *createIfThenRegion(
        Region         *head,
        Region         *then,
        Instruction    *terminator,
        bool           negated);

    /// Create a new IfThenElse region in the graph.
    ///
    /// \param head        the entry region
    /// \param then_node   the then region
    /// \param else_node   the else region
    /// \param terminator  the llvm basic block terminator instruction
    RegionIfThenElse *createIfThenElseRegion(
        Region         *head,
        Region         *then_node,
        Region         *else_node,
        Instruction    *terminator);

    /// Create a new natural loop region in the graph.
    ///
    /// \param head  the entry (and loop)
    RegionNaturalLoop *createNaturalLoopRegion(
        Region *head);

    /// Create a new Break region in the graph.
    ///
    /// \note a Break region is a kind of "edge label" and hence does not have a basic block
    RegionBreak *createBreakRegion();

    /// Create a new Continue region in the graph.
    ///
    /// \note a Continue region is a kind of "edge label" and hence does not have a basic block
    RegionContinue *createContinueRegion();

    /// Create a new Return region in the graph.
    ///
    /// \param head                the entry region of the Region
    /// \param return_instruction  the one and only return instruction inside this region
    RegionReturn *createReturnRegion(
        Region     *head,
        ReturnInst *return_inst);

    /// Create a new Switch region in the graph.
    ///
    /// \param head   the entry region of the Region
    /// \param cases  the ordered list description of the cases
    RegionSwitch *createSwitchRegion(
        Region                                       *head,
        ArrayRef<RegionSwitch::CaseDescriptor> const &cases);

    /// Delete a region.
    void dropRegion(Region *r);

    /// Recalculate the dominance tree for the function.
    void updateDomTree();

    /// Get the type mapper.
    mi::mdl::Type_mapper const &get_type_mapper() const { return m_type_mapper; }

private:
    // no copy
    StructuredFunction(StructuredFunction const &) = delete;
    // no assignment
    StructuredFunction &operator=(StructuredFunction const &) = delete;

private:
    /// The LLVM function.
    Function &m_func;

    /// The MDL type mapper for this function.
    mi::mdl::Type_mapper  &m_type_mapper;

    /// The dominator tree of the LLVM function.
    DominatorTree &m_domTree;

    /// The loop information of the LLVM function.
    LoopInfo &m_loop_info;

    /// The last used node ID.
    size_t m_node_id;

    /// The top level node containing the whole function.
    Region *m_body;

    std::map<BasicBlock *, Region *> m_mapping;

    /// The list of all Region nodes.
    RegionList m_region_list;
};

}  // sl
}  // llvm

#endif // MDL_GENERATOR_JIT_SL_FUNCTION_H
