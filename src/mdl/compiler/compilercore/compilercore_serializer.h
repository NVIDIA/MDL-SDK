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

#ifndef MDL_COMPILERCORE_SERIALIZER_H
#define MDL_COMPILERCORE_SERIALIZER_H 1

#include <mi/base/handle.h>
#include <mi/mdl/mdl_streams.h>
#include <mi/mdl/mdl_mdl.h>
#include <mi/mdl/mdl_serializer.h>

#include "compilercore_cc_conf.h"
#include "compilercore_allocator.h"
#include "compilercore_memory_arena.h"
#include "compilercore_assert.h"

#if 0
#define DOUT(x)     dprintf x
#define INC_SCOPE() dprintf_incscope()
#define DEC_SCOPE() dprintf_decscope()
#else
#define DOUT(x)
#define INC_SCOPE()
#define DEC_SCOPE()
#endif

namespace mi {
namespace mdl {

class ISymbol;
class Symbol_table;
class IType;
class IType_array_size;
class IType_factory;
class Type_factory;
class IValue;
class IValue_factory;
class Value_factory;
class IDefinition;
class Definition;
class Definition_table;
class IDeclaration;
class IQualified_name;
class ISimple_name;
class IType_name;
class IParameter;
class IExpression;
class IAnnotation_block;
class IAnnotation;
class IStatement;
class IArgument;
class MDL;
class Module;

/// The tag type.
typedef size_t Tag_t;

/// The invalid tag.
static const Tag_t INVALID_TAG = Tag_t(0);

/// Base class for the Pointer_serializer helper class that encodes pointers
/// to objects into Tag_t.
class Base_pointer_serializer
{
public:
    typedef ptr_hash_map<void, Tag_t>::Type::value_type     value_type;
    typedef ptr_hash_map<void, Tag_t>::Type::const_iterator const_iterator;

public:
    /// Constructor.
    ///
    /// \param alloc  the allocator
    explicit Base_pointer_serializer(IAllocator *alloc);

    /// returns true if the pointer map is empty.
    bool empty() const { return m_pointer_map.empty(); };

    /// Get the first element of the pointer map.
    const_iterator begin() const { return m_pointer_map.begin(); }

    /// Get the last element of the pointer map.
    const_iterator end() const { return m_pointer_map.end(); }

protected:
    /// Returns true if the given pointer is already known.
    ///
    /// \param obj  the object
    bool is_known_pointer(void *obj) const;

    /// Create a new tag for the given pointer.
    ///
    /// \param obj  the object
    Tag_t create_tag_for_pointer(void *obj);

    /// Get the tag for the given pointer.
    ///
    /// \param obj  the object
    ///
    /// \note obj must exist
    Tag_t get_tag_for_pointer(void *obj) const;

    /// Get or create the tag for the given pointer.
    ///
    /// \param obj  the object
    Tag_t get_or_create_tag_for_pointer(void *obj);

    /// Get the tag for the given pointer and delete the pointer from the pointer set.
    ///
    /// \param obj  the object
    ///
    /// \note obj must exist
    Tag_t get_tag_for_pointer_and_drop(void *obj);

private:
    typedef ptr_hash_map<void, Tag_t>::Type Pointer_map;

    /// The map storing pointer to tag translations.
    Pointer_map m_pointer_map;

    // The last assigned tag.
    Tag_t m_last_tag;
};

/// The Add_factory_ser mixin adds support for factory tags.
template<typename F>
class Add_factory_ser : public Base_pointer_serializer
{
    typedef Base_pointer_serializer Base;
public:
    /// Constructor.
    ///
    /// \param alloc  the allocator
    explicit Add_factory_ser(IAllocator *alloc)
    : Base(alloc)
    {
    }

    /// Create a new tag for the given factory pointer.
    Tag_t create_factory_tag(F const *obj) {
        return create_tag_for_pointer((void *)obj);
    }

    /// Get the tag for the given factory pointer.
    Tag_t get_factory_tag(F const *obj) const {
        return get_tag_for_pointer((void *)obj);
    }
};

/// The Pointer_serializer helper class encodes pointers
/// to objects into Tag_t.
template<typename T, typename F = Base_pointer_serializer>
class Pointer_serializer : public F
{
    typedef F Base;
public:
    /// Constructor.
    ///
    /// \param alloc  the allocator
    explicit Pointer_serializer(IAllocator *alloc)
    : Base(alloc)
    {
    }

    /// Returns true if the given object pointer is already known.
    ///
    /// \param obj  the object
    bool is_known(T const *obj) const {
        return Base::is_known_pointer((void *)obj);
    }

    /// Create a new tag for the given object pointer.
    Tag_t create_tag(T const *obj) {
        return Base::create_tag_for_pointer((void *)obj);
    }

    /// Get the tag for the given object pointer.
    Tag_t get_tag(T const *obj) const {
        return Base::get_tag_for_pointer((void *)obj);
    }

    /// Get the tag for the given object pointer and drop it.
    Tag_t get_tag_and_drop(T const *obj) {
        return Base::get_tag_for_pointer_and_drop((void *)obj);
    }

    /// Get or create the tag for the given object.
    Tag_t get_or_create_tag(T const *obj) {
        return Base::get_or_create_tag_for_pointer((void *)obj);
    }
};

/// The Pointer_deserializer helper class decodes tags
/// into pointer to objects.
class Base_pointer_deserializer
{
public:
    /// Constructor.
    ///
    /// \param alloc  the allocator
    explicit Base_pointer_deserializer(IAllocator *alloc);

    /// Returns true if the given tag is already known.
    ///
    /// \param tag  the tag
    bool is_known_tag(Tag_t tag) const;

protected:
    /// Register a pointer for a given tag.
    ///
    /// \param tag  the tag of the object
    /// \param obj  the object
    void register_pointer(Tag_t tag, void *obj);

    /// Get the pointer for a given tag.
    ///
    /// \param tag  the tag
    void *get_pointer(Tag_t tag) const;

private:
    typedef hash_map<Tag_t, void *>::Type Tag_map;

    /// The tag map.
    Tag_map m_tag_map;
};

/// The Add_factory_deser mixin adds support for factory tags.
template<typename F>
class Add_factory_deser : public Base_pointer_deserializer
{
    typedef Base_pointer_deserializer Base;
public:
    /// Constructor.
    ///
    /// \param alloc  the allocator
    explicit Add_factory_deser(IAllocator *alloc)
    : Base(alloc)
    {
    }

    /// Register a factory for a given tag.
    ///
    /// \param tag  the tag of the object
    /// \param obj  the factory
    void register_factory(Tag_t tag, F *obj) {
        register_pointer(tag, (void *)obj);
    }

    /// Get the factory pointer for a given tag.
    ///
    /// \param tag  the tag
    F *get_factory(Tag_t tag) const {
        return (F *)get_pointer(tag);
    }
};

/// The Pointer_deserializer helper class decodes tags
/// into pointer to objects.
template<typename T, typename F = Base_pointer_deserializer>
class Pointer_deserializer : public F
{
    typedef F Base;
public:
    /// Constructor.
    ///
    /// \param alloc  the allocator
    explicit Pointer_deserializer(IAllocator *alloc)
    : Base(alloc)
    {
    }

    /// Register a pointer for a given tag.
    ///
    /// \param tag  the tag of the object
    /// \param obj  the object
    void register_obj(Tag_t tag, T *obj) {
        Base::register_pointer(tag, (void *)obj);
    }

    /// Get the object pointer for a given tag.
    ///
    /// \param tag  the tag
    T *get_obj(Tag_t tag) const {
        return (T *)Base::get_pointer(tag);
    }
};

/// Base class for MDL object serializers.
class Base_serializer : public ISerializer
{
public:
    /// Write an int.
    ///
    /// \param v  the integer to write
    void write_int(int v) MDL_FINAL;

    /// Write a float.
    ///
    /// \param v  the float to write
    void write_float(float v) MDL_FINAL;

    /// Write a double.
    ///
    /// \param v  the double to write
    void write_double(double v) MDL_FINAL;

    /// Write an MDL section tag.
    ///
    /// \param tag  the MDL section tag to write.
    void write_section_tag(Serializer::Serializer_tags tag) MDL_FINAL;

    /// Write a (general) tag, assuming small values.
    ///
    /// \param tag  the tag to write
    void write_encoded_tag(size_t tag) MDL_FINAL;

    /// Write a c-string, supports NULL pointer.
    ///
    /// \param s  the string
    void write_cstring(char const *s) MDL_FINAL;

    /// Write a DB::Tag.
    ///
    /// \param tag  the DB::Tag encoded as 32bit
    void write_db_tag(unsigned tag) MDL_OVERRIDE;

    /// Destructor.
    virtual ~Base_serializer();
};

/// Base class for MDL object deserializers.
class Base_deserializer : public IDeserializer
{
public:
    /// Read an int.
    int read_int() MDL_FINAL;

    /// Read a float.
    float read_float() MDL_FINAL;

    /// Read a double.
    double read_double() MDL_FINAL;

    /// Read an MDL section tag.
    Serializer::Serializer_tags read_section_tag() MDL_FINAL;

    /// Read a (general) tag, assuming small values.
    size_t read_encoded_tag() MDL_FINAL;

    /// Read a c-string, supports NULL pointer.
    char const *read_cstring() MDL_FINAL;

    /// Reads a DB::Tag 32bit encoding.
    unsigned read_db_tag() MDL_OVERRIDE;

    /// Constructor.
    ///
    /// \param alloc  the allocator for temporary space
    Base_deserializer(IAllocator *alloc);

    /// Destructor.
    virtual ~Base_deserializer();

protected:
    IAllocator *m_alloc;

private:
    /// Temporary string buffer;
    char *m_string_buf;

    /// Length of the string buffer.
    size_t m_len;
};

/// Buffer serializer for MDL objects.
class Buffer_serializer : public Base_serializer
{
    typedef Base_serializer Base;
public:
    /// Write a byte.
    ///
    /// \param b  the byte to write
    void write(Byte b) MDL_FINAL;

    /// Get the data stream.
    Byte const *get_data() const;

    /// Get the size of the data.
    size_t get_size() const { return m_size; }

    /// Constructor.
    ///
    /// \param alloc  the allocator
    explicit Buffer_serializer(IAllocator *alloc);

    // Destructor.
    ~Buffer_serializer();

private:
    enum { LOAD_SIZE = 8 * 1024 };

    struct Header {
        size_t size;      ///< The size load section.
        Header *next;     ///< Point to the next header.

        Byte load[LOAD_SIZE];
    };

    /// The allocator used to allocate memory.
    IAllocator *m_alloc;

    /// Points to the first header.
    mutable Header *m_first;

    /// Points to the current header.
    mutable Header *m_curr;

    /// Points to the next byte written.
    mutable Byte *m_next;

    /// Points to the end byte of the current Header.
    mutable Byte *m_end;

    // Current size of the buffer.
    size_t m_size;
};

/// Buffer deserializer for MDL objects.
class Buffer_deserializer : public Base_deserializer
{
    typedef Base_deserializer Base;
public:
    /// Read a byte.
    Byte read() MDL_FINAL;

    /// Constructor.
    ///
    /// \param alloc  the allocator for temporary space
    /// \param data   the data stream
    /// \param size   the size of the data stream
    explicit Buffer_deserializer(
        IAllocator *alloc,
        Byte const *data,
        size_t     size);

    /// Destructor.
    ~Buffer_deserializer();

private:
    /// Points to the data stream.
    Byte const *m_data;

    // Points to the end of the data stream;
    Byte const *m_end;
};

/// A Serializer writing data to a stream.
class Stream_serializer : public Base_serializer {
public:
    typedef Base_serializer Base;
public:
    /// Write a byte.
    ///
    /// \param b  the byte to write
    void write(Byte b) MDL_FINAL;

    /// Constructor.
    ///
    /// \param os  an output stream
    explicit Stream_serializer(IOutput_stream *os);

private:
    /// The output stream.
    mi::base::Handle<IOutput_stream> m_os;
};

/// A Deserializer reading data from a stream.
class Stream_deserializer : public Base_deserializer {
public:
    typedef Base_deserializer Base;
public:
    /// Read a byte.
    Byte read() MDL_FINAL;

    /// Constructor.
    ///
    /// \param alloc  the allocator
    /// \param is     an input stream
    explicit Stream_deserializer(IAllocator *alloc, IInput_stream *is);

private:
    /// The input stream.
    mi::base::Handle<IInput_stream> m_is;
};

/// Base class for Binary and Module serializer.
class Entity_serializer {
public:
    /// Get the allocator of this serializer.
    IAllocator *get_allocator() const { return m_alloc; }

    /// Write an MDL section tag.
    ///
    /// \param tag  the MDL section tag to write.
    void write_section_tag(Serializer::Serializer_tags tag) {
        m_serializer->write_section_tag(tag);
    }

    /// Write a (general) tag, assuming small values.
    ///
    /// \param tag  the tag to write
    void write_encoded_tag(size_t tag) {
        m_serializer->write_encoded_tag(tag);
    }

    /// Write a c-string, supports NULL pointer.
    ///
    /// \param s  the string
    void write_cstring(char const *s) {
        m_serializer->write_cstring(s);
    }

    /// Write a byte.
    ///
    /// \param b  the value
    void write_byte(unsigned char b) {
        m_serializer->write(b);
    }

    /// Write a boolean value.
    ///
    /// \param b  the value
    void write_bool(bool b) {
        m_serializer->write(b);
    }

    /// Write an unsigned value.
    ///
    /// \param v  the value
    void write_unsigned(unsigned v) {
        m_serializer->write_encoded_tag(v);
    }

    /// Write an integer value, allowing negative values.
    ///
    /// \param v  the value
    void write_int(int v) {
        m_serializer->write_int(v);
    }

    /// Write a float value.
    ///
    /// \param v  the value
    void write_float(float v) {
        m_serializer->write_float(v);
    }

    /// Write a double value.
    ///
    /// \param v  the value
    void write_double(double v) {
        m_serializer->write_double(v);
    }

    /// Write a DB::Tag.
    ///
    /// \param tag  the DB::Tag encoded as 32bit
    void write_db_tag(unsigned tag) {
        m_serializer->write_db_tag(tag);
    }

    /// Constructor.
    ///
    /// \param alloc       the allocator
    /// \param serializer  the serializer used to write the low level data.
    Entity_serializer(
        IAllocator  *alloc,
        ISerializer *serializer);

protected:
    /// The allocator interface.
    IAllocator *m_alloc;

    /// The low level serializer.
    ISerializer *m_serializer;
};

/// The MDL binary serializer.
class MDL_binary_serializer : public Entity_serializer {
    typedef hash_map<size_t, Module const *>::Type Id_map;
public:
    /// Check if the given module is known by this binary serializer.
    ///
    /// \param mod  the module
    bool is_known_module(Module const *mod) const;

    /// Register a module.
    ///
    /// \param mod  the module
    Tag_t register_module(Module const *mod);

    /// Get the tag for a known module.
    ///
    /// \param mod  the module
    Tag_t get_module_tag(Module const *mod) const;

    /// Get the tag for a known module ID.
    ///
    /// \param id  the module ID
    Tag_t get_module_tag(size_t id) const;

    /// Check if a given module was already registered.
    ///
    /// \param mod  the module
    bool is_module_registered(Module const *mod) const
    {
        return m_modules.is_known(mod);
    }

    /// Constructor.
    ///
    /// \param alloc       the allocator
    /// \param compiler    the compiler
    /// \param serializer  the serializer used to write the low level data.
    MDL_binary_serializer(
        IAllocator  *alloc,
        MDL const   *compiler,
        ISerializer *serializer);

private:
    /// pointer serializer for imported modules.
    Pointer_serializer<Module> m_modules;

    /// Maps original module IDs to modules.
    Id_map m_id_map;
};

/// The MDL factory serializer.
class Factory_serializer : public Entity_serializer {
public:
    /// Register a symbol.
    ///
    /// \param sym  the symbol
    Tag_t register_symbol(ISymbol const *sym) {
        return m_symbols.create_tag(sym);
    }

    /// Get the tag for a known Symbol.
    ///
    /// \param sym  the symbol
    Tag_t get_symbol_tag(ISymbol const *sym) {
        return m_symbols.get_tag(sym);
    }

    /// Register a symbol table.
    ///
    /// \param symtab  the symbol table
    Tag_t register_symbol_table(Symbol_table const *symtab)
    {
        return m_symbols.create_factory_tag(symtab);
    }

    /// Get the tag for a known symbol table.
    ///
    /// \param symtab  the symbol table
    Tag_t get_symbol_table_tag(Symbol_table const *symtab)
    {
        return m_symbols.get_factory_tag(symtab);
    }

    /// Register a type.
    ///
    /// \param type  the type
    Tag_t register_type(IType const *type) {
        return m_types.create_tag(type);
    }

    /// Get the tag for a known type.
    ///
    /// \param type  the type
    Tag_t get_type_tag(IType const *type)
    {
        Tag_t tag = m_types.get_tag(type);
        MDL_ASSERT(tag != 0);
        return tag;
    }

    /// Register a type factory.
    ///
    /// \param tf  the type factory
    Tag_t register_type_factory(IType_factory const *tf)
    {
        return m_types.create_factory_tag(tf);
    }

    /// Get the tag for a known type factory.
    ///
    /// \param tf  the type factory
    Tag_t get_type_factory_tag(IType_factory const *tf)
    {
        return m_types.get_factory_tag(tf);
    }

    /// Register a deferred array size.
    ///
    /// \param array_size  the array size
    Tag_t register_array_size(IType_array_size const *array_size) {
        return m_array_sizes.create_tag(array_size);
    }

    /// Get the tag for a known deferred array size.
    ///
    /// \param array_size  the array size
    Tag_t get_array_size_tag(IType_array_size const *array_size)
    {
        return m_array_sizes.get_tag(array_size);
    }

    /// Register a value.
    ///
    /// \param v  the value
    Tag_t register_value(IValue const *v) {
        return m_values.create_tag(v);
    }

    /// Get the tag for a known value.
    ///
    /// \param v  the value
    Tag_t get_value_tag(IValue const *v)
    {
        Tag_t tag = m_values.get_tag(v);
        MDL_ASSERT(tag != 0);
        return tag;
    }

    /// Register a value factory.
    ///
    /// \param vf  the value factory
    Tag_t register_value_factory(IValue_factory const *vf)
    {
        return m_values.create_factory_tag(vf);
    }

    /// Get the tag for a known value factory.
    ///
    /// \param vf  the value factory
    Tag_t get_value_factory_tag(IValue_factory const *vf)
    {
        return m_values.get_factory_tag(vf);
    }

    /// Register a module.
    ///
    /// \param mod  the module
    Tag_t register_module(Module const *mod)
    {
        return m_bin_serializer->register_module(mod);
    }

    /// Check if the given module is known.
    ///
    /// \param mod  the module
    bool is_known_module(Module const *mod) const
    {
        return m_bin_serializer->is_known_module(mod);
    }

    /// Get the tag for a known module.
    ///
    /// \param mod  the module
    Tag_t get_module_tag(Module const *mod) const
    {
        return m_bin_serializer->get_module_tag(mod);
    }

    /// Get the tag for a known module ID.
    ///
    /// \param id  the module ID
    Tag_t get_module_tag(size_t id) const
    {
        return m_bin_serializer->get_module_tag(id);
    }

    /// Check if a given module was already registered.
    ///
    /// \param mod  the module
    bool is_module_registered(Module const *mod) const
    {
        return m_bin_serializer->is_module_registered(mod);
    }

    /// Enqueue the given type for processing.
    ///
    /// \param type  the type
    void enqueue_type(IType const *type);

    /// Write all enqueued types.
    void write_enqueued_types();

    /// Enqueue the given value for processing.
    ///
    /// \param v  the value
    void enqueue_value(IValue const *v);

    /// Write all enqueued values.
    void write_enqueued_values();

    /// Write an AST declaration.
    ///
    /// \param decl  the declaration
    void write_decl(IDeclaration const *decl);

    /// Write an AST simple name.
    ///
    /// \param sname  the simple name
    void write_name(ISimple_name const *sname);

    /// Write an AST qualified name.
    ///
    /// \param qname  the qualified name
    void write_name(IQualified_name const *qname);

    /// Write an AST type name.
    ///
    /// \param tname  the type name
    void write_name(IType_name const *tname);

    /// Write an AST parameter.
    ///
    /// \param param  the parameter
    void write_parameter(IParameter const *param);

    /// Write an AST expression.
    ///
    /// \param expr  the expression
    void write_expr(IExpression const *expr);

    /// Write an AST annotation block.
    ///
    /// \param block  the annotation block
    void write_annos(IAnnotation_block const *block);

    /// Write an AST annotation.
    ///
    /// \param anno  the annotation
    void write_anno(IAnnotation const *anno);

    /// Write an AST statement.
    ///
    /// \param stmt  the statement
    void write_stmt(IStatement const *stmt);

    /// Write an AST argument.
    ///
    /// \param arg  the argument
    void write_argument(IArgument const *arg);

    /// Write an AST position.
    ///
    /// \param pos  the position
    void write_pos(Position const *pos);

    /// Write all initializer expressions unreferenced so far.
    void write_unreferenced_init_expressions();

    /// Constructor.
    ///
    /// \param alloc           the allocator
    /// \param serializer      the serializer used to write the low level data
    /// \param bin_serializer  the serializer used for serializing "the binary"
    Factory_serializer(
        IAllocator            *alloc,
        ISerializer           *serializer,
        MDL_binary_serializer *bin_serializer);

private:
    /// Put the given type into the type wait queue.
    ///
    /// \param type  the type
    void push_type(IType const *type);

    /// Mark a type as used by an compound type.
    ///
    /// \param type  the type
    void mark_child_type(IType const *type);

    /// Process a type and its sub-types in DFS order.
    ///
    /// \param type           the type
    /// \param is_child_type  if true, this type is a sub-type of another type
    void dfs_type(IType const *type, bool is_child_type = true);

    /// Write the given type.
    ///
    /// \param type  the type
    void write_type(IType const *type);

    /// Put the given value into the value wait queue.
    ///
    /// \param value  the value
    void push_value(IValue const *value);

    /// Mark a value as used by an compound type.
    ///
    /// \param type  the type
    void mark_child_value(IValue const *value);

    /// Process a value and its sub-values in DFS order.
    ///
    /// \param value           the value
    /// \param is_child_value  if true, this value is a sub-value of another compound value
    void dfs_value(IValue const *value, bool is_child_value = true);

    /// Write the given value.
    ///
    /// \param v  the value
    void write_value(IValue const *v);

private:
    /// pointer serializer for the symbol table.
    Pointer_serializer<ISymbol, Add_factory_ser<Symbol_table> > m_symbols;

    /// pointer serializer for the type table.
    Pointer_serializer<IType, Add_factory_ser<IType_factory> > m_types;

    /// pointer serializer for deferred array sizes.
    Pointer_serializer<IType_array_size> m_array_sizes;

    /// pointer serializer for the value table.
    Pointer_serializer<IValue, Add_factory_ser<IValue_factory> > m_values;

    typedef ptr_hash_set<IType const>::Type Type_set;
    typedef vector<IType const *>::Type     Type_queue;

    /// The visited set for types.
    Type_set m_types_visited;

    /// The set of child types.
    Type_set m_child_types;

    /// The estimated type root set.
    Type_queue m_type_root_queue;

    /// The type queue.
    Type_queue m_type_queue;

    typedef ptr_hash_set<IValue const>::Type Value_set;
    typedef vector<IValue const *>::Type     Value_queue;

    /// The visited set for values.
    Value_set m_values_visited;

    /// The set of child values.
    Value_set m_child_values;

    /// The estimated value root set.
    Value_queue m_value_root_queue;

    /// The value queue.
    Value_queue m_value_queue;

    /// If true, the root set building phase is active.
    bool m_building_root_set;

    /// The serializer of the "binary" if any.
    MDL_binary_serializer *m_bin_serializer;

#ifdef ENABLE_ASSERT
    /// Helper set to mark those values already in the value queue.
    Value_set m_value_queue_marker;
#endif
};

/// The MDL module serializer.
class Module_serializer : public Factory_serializer {
public:
    /// Register a definition.
    ///
    /// \param def  the definition
    Tag_t register_definition(IDefinition const *def) {
        // for definitions we allow multiple registering, because
        // inside the definition table a definition might point to a later
        // definition by the "definite" property.
        return m_definitions.get_or_create_tag(def);
    }

    /// Get the tag for a known definition.
    ///
    /// \param def  the definition
    Tag_t get_definition_tag(IDefinition const *def)
    {
        Tag_t tag = m_definitions.get_tag(def);
        MDL_ASSERT(tag != 0);
        return tag;
    }

    /// Register a definition table.
    ///
    /// \param dt  the definition table
    Tag_t register_definition_table(Definition_table const *dt)
    {
        return m_definitions.create_factory_tag(dt);
    }

    /// Get the tag for a known definition table.
    ///
    /// \param dt  the definition table
    Tag_t get_definition_factory_tag(Definition_table const *dt)
    {
        return m_definitions.get_factory_tag(dt);
    }

    /// Register a declaration.
    ///
    /// \param decl  the declaration
    Tag_t register_declaration(IDeclaration const *decl) {
        // for declaration we allow multiple registering, because
        // they occur in the definition table BEFORE we dumped them
        // with the AST.
        return m_declaratios.get_or_create_tag(decl);
    }

    /// Get the tag for a known declaration.
    ///
    /// \param decl  the declaration
    Tag_t get_declaration_tag(IDeclaration const *decl)
    {
        Tag_t tag = m_declaratios.get_tag(decl);
        MDL_ASSERT(tag != 0);
        return tag;
    }

    /// Register an expression.
    ///
    /// \param expr  the expression
    Tag_t register_expression(IExpression const *expr) {
        return m_init_exprs.get_or_create_tag(expr);
    }

    /// Get the tag for a known expression or return Tag_t(0).
    ///
    /// \param expr  the expression
    Tag_t get_expression_tag_and_drop(IExpression const *expr)
    {
        if (m_init_exprs.is_known(expr)) {
            Tag_t tag = m_init_exprs.get_tag_and_drop(expr);
            MDL_ASSERT(tag != 0);
            return tag;
        }
        return Tag_t(0);
    }

    /// Write an AST declaration.
    ///
    /// \param decl  the declaration
    void write_decl(IDeclaration const *decl);

    /// Write an AST simple name.
    ///
    /// \param sname  the simple name
    void write_name(ISimple_name const *sname);

    /// Write an AST qualified name.
    ///
    /// \param qname  the qualified name
    void write_name(IQualified_name const *qname);

    /// Write an AST type name.
    ///
    /// \param tname  the type name
    void write_name(IType_name const *tname);

    /// Write an AST parameter.
    ///
    /// \param param  the parameter
    void write_parameter(IParameter const *param);

    /// Write an AST expression.
    ///
    /// \param expr  the expression
    void write_expr(IExpression const *expr);

    /// Write an AST annotation block.
    ///
    /// \param block  the annotation block
    void write_annos(IAnnotation_block const *block);

    /// Write an AST annotation.
    ///
    /// \param anno  the annotation
    void write_anno(IAnnotation const *anno);

    /// Write an AST statement.
    ///
    /// \param stmt  the statement
    void write_stmt(IStatement const *stmt);

    /// Write an AST argument.
    ///
    /// \param arg  the argument
    void write_argument(IArgument const *arg);

    /// Write an AST position.
    ///
    /// \param pos  the position
    void write_pos(Position const *pos);

    /// Write all initializer expressions unreferenced so far.
    void write_unreferenced_init_expressions();

    /// Constructor.
    ///
    /// \param alloc           the allocator
    /// \param serializer      the serializer used to write the low level data
    /// \param bin_serializer  the serializer used for serializing "the binary"
    Module_serializer(
        IAllocator            *alloc,
        ISerializer           *serializer,
        MDL_binary_serializer *bin_serializer);

private:
    /// pointer serializer for the definition table.
    Pointer_serializer<IDefinition, Add_factory_ser<Definition_table> > m_definitions;

    /// pointer serializer for declarations.
    Pointer_serializer<IDeclaration> m_declaratios;

    /// pointer serializer for initializer expressions.
    Pointer_serializer<IExpression> m_init_exprs;
};

/// Base class for the wait queue for entities not yet serialized.
class Base_wait_queue {
    struct Entry {
        void  **m_dst;    ///< The destination address.
        Entry *m_next;    ///< Points to the next entry.

        explicit Entry(void **adr) : m_dst(adr), m_next(NULL) {}
    };

public:
    /// Constructor.
    ///
    /// \param alloc  the memory allocator to be used
    explicit Base_wait_queue(IAllocator *alloc);

    /// Destructor.
    ~Base_wait_queue();

    /// Wait for the given tag.
    ///
    /// \param tag  the tag to wait for
    /// \param adr  the address where the object of type T must be entered
    void wait(Tag_t tag, void **adr);

    /// Object gets ready.
    ///
    /// \param tag  the tag of the object
    /// \param obj  the object itself
    ///
    /// Enters obj at all addresses that waits for it.
    void ready(Tag_t tag, void *obj);

private:
    /// Get a new entry for a given destination address.
    ///
    /// \param adr  the destination address
    Entry *get_entry(void **adr);

private:
    /// The allocator.
    IAllocator *m_alloc;

    /// The builder for objects.
    Allocator_builder m_builder;

    /// List of free wait entries.
    Entry *m_free_list;

    typedef map<Tag_t, Entry *>::Type Wait_lists;

    /// The wait lists.
    Wait_lists m_wait_list;
};

/// Type safe wait queue for entities not yet serialized.
template<typename T>
class Entity_wait_queue : protected Base_wait_queue {
    typedef Base_wait_queue Base;
public:
    /// Constructor.
    ///
    /// \param alloc  the memory allocator to be used
    explicit Entity_wait_queue(IAllocator *alloc)
    : Base(alloc)
    {
    }

    /// Wait for the given tag.
    ///
    /// \param tag  the tag to wait for
    /// \param adr  the address where the object of type T must be entered
    void wait(Tag_t tag, T **adr) { Base::wait(tag, (void **)adr); }

    /// Object gets ready.
    ///
    /// \param tag  the tag of the object
    /// \param obj  the object itself
    ///
    /// Enters obj at all addresses that waits for it.
    void ready(Tag_t tag, T *obj) { Base::ready(tag, (void *)obj); }
};

/// Base class for Binary and Module serializer.
class Entity_deserializer {
public:
    /// Read an MDL section tag.
    Serializer::Serializer_tags read_section_tag() {
        return m_deserializer->read_section_tag();
    }

    /// Read a (general) tag, assuming small values.
    size_t read_encoded_tag() {
        return m_deserializer->read_encoded_tag();
    }

    /// Read a c-string, supports NULL pointer.
    char const *read_cstring() {
        return m_deserializer->read_cstring();
    }

    /// Read a byte.
    unsigned char read_byte() {
        return m_deserializer->read();
    }

    /// Read a boolean value.
    bool read_bool() {
        unsigned char b = m_deserializer->read();
        MDL_ASSERT(b == 0 || b == 1);
        return b != 0;
    }

    /// Read an size_t value.
    size_t read_size_t() {
        return m_deserializer->read_encoded_tag();
    }

    /// Read an unsigned value.
    unsigned read_unsigned() {
        return unsigned(m_deserializer->read_encoded_tag());
    }

    /// Read an integer value.
    int read_int() {
        return m_deserializer->read_int();
    }

    /// Read a float value.
    float read_float() {
        return m_deserializer->read_float();
    }

    /// Read a double value.
    double read_double() {
        return m_deserializer->read_double();
    }

    /// Reads a DB::Tag 32bit encoding.
    unsigned read_db_tag() {
        return m_deserializer->read_db_tag();
    }

    /// Get the allocator.
    IAllocator *get_allocator() const { return m_alloc; }

    /// Constructor.
    ///
    /// \param alloc         the allocator
    /// \param compiler      the compiler
    /// \param deserializer  the deserializer used to write the low level data.
    Entity_deserializer(
        IAllocator    *alloc,
        IDeserializer *deserializer);

protected:
    /// The allocator.
    IAllocator *m_alloc;

    /// The low level serializer.
    IDeserializer *m_deserializer;
};

/// The MDL binary deserializer.
class MDL_binary_deserializer : public Entity_deserializer {
public:
    /// Register a module.
    ///
    /// \param tag   the module tag
    /// \param v     the module
    void register_module(Tag_t tag, Module const *mod) {
        m_modules.register_obj(tag, mod);
    }

    /// Get the module for for a known tag.
    ///
    /// \param tag     the module tag
    Module const *get_module(Tag_t tag) const
    {
        return m_modules.get_obj(tag);
    }

    /// Check if a given module was already registered.
    ///
    /// \param tag  the module tag
    bool is_known_module(Tag_t tag) const
    {
        return m_modules.is_known_tag(tag);
    }

    /// Constructor.
    ///
    /// \param alloc         the allocator
    /// \param deserializer  the deserializer used to write the low level data.
    /// \param compiler      the compiler
    MDL_binary_deserializer(
        IAllocator    *alloc,
        IDeserializer *deserializer,
        MDL           *compiler);

private:
    /// interface deserializer for modules.
    Pointer_deserializer<Module const> m_modules;
};

/// The MDL factory deserializer.
class Factory_deserializer : public Entity_deserializer {
public:
    /// Register a symbol.
    ///
    /// \param tag  the symbol tag
    /// \param sym  the symbol
    void register_symbol(Tag_t tag, ISymbol const *sym) {
        m_symbols.register_obj(tag, sym);
    }

    /// Get the symbol for for a known tag.
    ///
    /// \param tag     the symbol tag
    ISymbol const *get_symbol(Tag_t tag)
    {
        return m_symbols.get_obj(tag);
    }

    /// Register a symbol table.
    ///
    /// \param tag     the symbol table tag
    /// \param symtab  the symbol table
    void register_symbol_table(Tag_t tag, Symbol_table const *symtab)
    {
        m_symbols.register_factory(tag, symtab);
    }

    /// Get the symbol table for for a known tag.
    ///
    /// \param tag     the symbol table tag
    Symbol_table const *get_symbol_table(Tag_t tag)
    {
        return m_symbols.get_factory(tag);
    }

    /// Register a type.
    ///
    /// \param tag   the type tag
    /// \param type  the type
    void register_type(Tag_t tag, IType const *type) {
        m_types.register_obj(tag, type);
    }

    /// Get the type for for a known tag.
    ///
    /// \param tag     the type tag
    IType const *get_type(Tag_t tag)
    {
        return m_types.get_obj(tag);
    }

    /// Register a type factory.
    ///
    /// \param tag   the type factory tag
    /// \param tf    the type factory
    void register_type_factory(Tag_t tag, IType_factory const *tf) {
        m_types.register_factory(tag, tf);
    }

    /// Get the type factory for for a known tag.
    ///
    /// \param tag     the type factory tag
    IType_factory const *get_type_factory(Tag_t tag)
    {
        return m_types.get_factory(tag);
    }

    /// Register a deferred array size.
    ///
    /// \param tag         the array size tag
    /// \param array_size  the array size
    void register_array_size(Tag_t tag, IType_array_size const *array_size) {
        m_array_sizes.register_obj(tag, array_size);
    }

    /// Get the array size for for a known tag.
    ///
    /// \param tag     the array size tag
    IType_array_size const *get_array_size(Tag_t tag)
    {
        return m_array_sizes.get_obj(tag);
    }

    /// Register a value.
    ///
    /// \param tag   the value tag
    /// \param v     the value
    void register_value(Tag_t tag, IValue const *v) {
        m_values.register_obj(tag, v);
    }

    /// Get the value for for a known tag.
    ///
    /// \param tag     the value tag
    IValue const *get_value(Tag_t tag)
    {
        return m_values.get_obj(tag);
    }

    /// Register a value factory.
    ///
    /// \param tag   the value factory tag
    /// \param vf    the value factory
    void register_value_factory(Tag_t tag, IValue_factory const *vf) {
        m_values.register_factory(tag, vf);
    }

    /// Get the value factory for for a known tag.
    ///
    /// \param tag     the value factory tag
    IValue_factory const *get_value_factory(Tag_t tag)
    {
        return m_values.get_factory(tag);
    }

    /// Register a module.
    ///
    /// \param tag   the module tag
    /// \param v     the module
    void register_module(Tag_t tag, Module const *mod)
    {
        MDL_ASSERT(m_bin_deserializer != NULL && "not in binary mode");
        m_bin_deserializer->register_module(tag, mod);
    }

    /// Get the module for for a known tag.
    ///
    /// \param tag     the module tag
    Module const *get_module(Tag_t tag) const
    {
        MDL_ASSERT(m_bin_deserializer != NULL && "not in binary mode");
        return m_bin_deserializer->get_module(tag);
    }

    /// Check if a given module was already registered.
    ///
    /// \param tag  the module tag
    bool is_known_module(Tag_t tag) const
    {
        MDL_ASSERT(m_bin_deserializer != NULL && "not in binary mode");
        return m_bin_deserializer->is_known_module(tag);
    }

    /// Read a type.
    ///
    /// \param tf  the type factory used to restore the type
    IType const *read_type(Type_factory &tf);

    /// Read a value.
    ///
    /// \param vf  the value factory used to restore the value
    IValue const *read_value(Value_factory &vf);

    /// Constructor.
    ///
    /// \param alloc             the allocator
    /// \param deserializer      the deserializer used to read the low level data
    /// \param bin_deserializer  the serializer used for deserializing "the binary"
    Factory_deserializer(
        IAllocator              *alloc,
        IDeserializer           *deserializer,
        MDL_binary_deserializer *bin_deserializer);

private:
    /// pointer deserializer for the symbol table.
    Pointer_deserializer<ISymbol const, Add_factory_deser<Symbol_table const> > m_symbols;

    /// pointer deserializer for the type table.
    Pointer_deserializer<IType const, Add_factory_deser<IType_factory const> > m_types;

    /// pointer deserializer for deferred array sizes.
    Pointer_deserializer<IType_array_size const> m_array_sizes;

    /// pointer deserializer for the value table.
    Pointer_deserializer<IValue const, Add_factory_deser<IValue_factory const> > m_values;

    /// The "binary" deserializer if any.
    MDL_binary_deserializer *m_bin_deserializer;
};

/// The MDL module deserializer.
class Module_deserializer : public Factory_deserializer {
public:
    /// Register a definition.
    ///
    /// \param tag   the definition tag
    /// \param def   the definition
    void register_definition(Tag_t tag, Definition *def) {
        m_definitions.register_obj(tag, def);
        m_def_wait_queue.ready(tag, def);
    }

    /// Get the definition for for a known tag.
    ///
    /// \param tag     the definition tag
    Definition *get_definition(Tag_t tag)
    {
        return m_definitions.get_obj(tag);
    }

    /// Register a definition table.
    ///
    /// \param tag   the definition table tag
    /// \param dt    the definition table
    void register_definition_table(Tag_t tag, Definition_table const *dt) {
        m_definitions.register_factory(tag, dt);
    }

    /// Get the definition table for for a known tag.
    ///
    /// \param tag     the definition factory tag
    Definition_table const *get_definition_table(Tag_t tag)
    {
        return m_definitions.get_factory(tag);
    }

    /// Register a declaration.
    ///
    /// \param tag   the declaration tag
    /// \param decl  the declaration
    void register_declaration(Tag_t tag, IDeclaration const *decl) {
        m_declarations.register_obj(tag, decl);
        m_decl_wait_queue.ready(tag, decl);
    }

    /// Get the declaration for for a known tag.
    ///
    /// \param tag     the declaration tag
    IDeclaration const *get_declaration(Tag_t tag)
    {
        return m_declarations.get_obj(tag);
    }

    /// Register an expression.
    ///
    /// \param tag   the expression tag
    /// \param expr  the expression
    void register_expression(Tag_t tag, IExpression *expr) {
        m_init_exprs.register_obj(tag, expr);
        m_expr_wait_queue.ready(tag, expr);
    }

    /// Get the expression for for a known tag.
    ///
    /// \param tag     the expression tag
    IExpression *get_expression(Tag_t tag)
    {
        return m_init_exprs.get_obj(tag);
    }

    /// The given location waits for a declaration to become ready.
    ///
    /// \param tag  a declaration tag
    /// \param loc  a location
    void wait_for_declaration(Tag_t tag, IDeclaration const **loc);

    /// The given location waits for a definition to become ready.
    ///
    /// \param tag  a definition tag
    /// \param loc  a location
    void wait_for_definition(Tag_t tag, Definition **loc);

    /// The given location waits for an expression to become ready.
    ///
    /// \param tag  an expression tag
    /// \param loc  a location
    void wait_for_expression(Tag_t tag, IExpression const **loc);

    /// Read an AST declaration.
    ///
    /// \param mod  the owner Module
    ///
    /// \return the read declaration
    IDeclaration *read_decl(Module &mod);

    /// Read all initializer expressions unreferenced so far.
    ///
    /// \param mod  the owner Module
    void read_unreferenced_init_expressions(Module &mod);

    /// Creates a new (empty) module for deserialization.
    ///
    /// \param mdl_version  the MDL language level of the module
    /// \param analyzed     true, if an analyzed module will be deserialized
    ///
    /// \return a new empty module
    Module *create_module(
        IMDL::MDL_version mdl_version,
        bool              analyzed);

    /// Constructor.
    ///
    /// \param alloc             the allocator
    /// \param deserializer      the deserializer used to write the low level data
    /// \param bin_deserializer  the serializer used for deserializing "the binary"
    /// \param compiler          the compiler
    Module_deserializer(
        IAllocator              *alloc,
        IDeserializer           *deserializer,
        MDL_binary_deserializer *bin_deserializer,
        MDL                     *compiler);

private:
    /// Read an AST simple name.
    ///
    /// \param mod  the owner Module
    ///
    /// \return the read simple name
    ISimple_name const *read_sname(Module &mod);

    /// Read an AST qualified name.
    ///
    /// \param mod  the owner Module
    ///
    /// \return the read qualified name
    IQualified_name *read_qname(Module &mod);

    /// Read an AST type name.
    ///
    /// \param mod  the owner Module
    ///
    /// \return the read type name
    IType_name const *read_tname(Module &mod);

    /// Read an AST parameter.
    ///
    /// \param mod  the owner Module
    ///
    /// \return the read parameter
    IParameter const *read_parameter(Module &mod);

    /// Read an AST expression.
    ///
    /// \param mod  the owner Module
    ///
    /// \return the read expression
    IExpression *read_expr(Module &mod);

    /// Read an AST annotation block.
    ///
    /// \param mod  the owner Module
    ///
    /// \return the read annotation block or NULL
    IAnnotation_block const *read_annos(Module &mod);

    /// Read an AST annotation.
    ///
    /// \param mod  the owner Module
    ///
    /// \return the read annotation
    IAnnotation const *read_anno(Module &mod);

    /// Read an AST statement.
    ///
    /// \param mod  the owner Module
    ///
    /// \return the read statement
    IStatement const *read_stmt(Module &mod);

    /// Read an AST argument.
    ///
    /// \param mod  the owner Module
    ///
    /// \return the read argument
    IArgument const *read_argument(Module &mod);

    /// Read an AST position.
    ///
    /// \param pos  the position to read
    void read_pos(Position &pos);

private:
    /// The compiler that will own the created modules.
    MDL *m_compiler;

    /// pointer deserializer for the definition table.
    Pointer_deserializer<Definition, Add_factory_deser<Definition_table const> > m_definitions;

    /// pointer deserializer for declarations.
    Pointer_deserializer<IDeclaration const> m_declarations;

    /// pointer deserializer for initializer expressions.
    Pointer_deserializer<IExpression> m_init_exprs;

    /// Wait queue for declarations.
    Entity_wait_queue<IDeclaration const> m_decl_wait_queue;

    /// Wait queue for definitions.
    Entity_wait_queue<Definition> m_def_wait_queue;

    /// Wait queue for expressions.
    Entity_wait_queue<IExpression const> m_expr_wait_queue;
};

void dprintf(char const *fmt, ...);
void dprintf_incscope();
void dprintf_decscope();

}  // mdl
}  // mi

#endif
