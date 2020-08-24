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

#include "command_queue.h"

namespace mi { namespace examples { namespace mdl_d3d12
{

namespace {

static const GUID s_command_list_pd_queue =
    {0x67a0fd22, 0x7dcb, 0x4bde, { 0xbc, 0x6e, 0xb2, 0x5f, 0x79, 0x59, 0x96, 0x1c }};

static const GUID s_command_list_pd_allocator =
    {0xe82371a0, 0x9b9c, 0x40e7, { 0xbb, 0xf5, 0x6, 0xf8, 0x8d, 0xe, 0xc5, 0x87 }};

} // anonymous


Command_queue::Command_queue(Base_application* app, D3D12_COMMAND_LIST_TYPE type)
    : m_app(app)
    , m_command_list_type(type)
    , m_fence(nullptr)
    , m_mtx()
{
    D3D12_COMMAND_QUEUE_DESC desc = {};
    desc.Type = m_command_list_type;
    desc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
    desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    desc.NodeMask = 0;

    throw_on_failure(app->get_device()->CreateCommandQueue(
        &desc, IID_PPV_ARGS(&m_command_queue)),
        "Failed to create command queue.", SRC);

    m_fence = new Fence(app, this);
}

// ------------------------------------------------------------------------------------------------

Command_queue::~Command_queue()
{
    m_mtx.lock();
    m_all_command_allocators.clear();
    while (!m_command_allocator_queue.empty())
        m_command_allocator_queue.pop();

    m_all_command_lists.clear();
    while (!m_free_command_list_queue.empty())
        m_free_command_list_queue.pop();

    if (m_fence)
        delete m_fence;

    m_mtx.unlock();
}

// ------------------------------------------------------------------------------------------------

D3DCommandList* Command_queue::get_command_list()
{
    ID3D12CommandAllocator* command_allocator = nullptr;
    D3DCommandList* command_list = nullptr;

    m_mtx.lock();
    if (!m_command_allocator_queue.empty() &&
        m_fence->is_completed(m_command_allocator_queue.front().handle))
    {
        command_allocator = m_command_allocator_queue.front().allocator;
        m_command_allocator_queue.pop();
        throw_on_failure(command_allocator->Reset(),
            "Failed to reset command allocator", SRC);
    }
    else
    {
        ComPtr<ID3D12CommandAllocator> new_allocator = create_command_allocator();
        m_all_command_allocators.push_back(new_allocator);
        command_allocator = new_allocator.Get();
    }

    if (!m_free_command_list_queue.empty())
    {
        command_list = m_free_command_list_queue.front();
        m_free_command_list_queue.pop();
        throw_on_failure(command_list->Reset(command_allocator, nullptr),
            "Failed to reset command list", SRC);
    }
    else
    {
        ComPtr<D3DCommandList> new_list = create_command_list(command_allocator);
        m_all_command_lists.push_back(new_list);
        command_list = m_all_command_lists.back().Get();
    }
    m_mtx.unlock();

    ID3D12CommandAllocator* pd_allocator[1] = {command_allocator};
    command_list->SetPrivateData(
        s_command_list_pd_allocator, sizeof(ID3D12CommandAllocator*), pd_allocator);

    Command_queue* pd_queue[1] = { this };
    command_list->SetPrivateData(
        s_command_list_pd_queue, sizeof(Command_queue*), pd_queue);

    return command_list;
}

// ------------------------------------------------------------------------------------------------

UINT64 Command_queue::execute_command_list(D3DCommandList* command_list)
{
    throw_on_failure(command_list->Close(), "Failed to close command list", SRC);

    UINT size = sizeof(Command_queue*);
    Command_queue* queue = nullptr;
    HRESULT res = command_list->GetPrivateData(s_command_list_pd_queue, &size, &queue);
    if (res != S_OK || queue != this)
    {
        std::string message = "Command list can only be executed on the queue that created it.";
        log_error(message, SRC);
        throw(message);
    }

    size = sizeof(ID3D12CommandAllocator*);
    ID3D12CommandAllocator* allocator = nullptr;
    res = command_list->GetPrivateData(s_command_list_pd_allocator, &size, &allocator);

    ID3D12CommandList* const ppCommandLists[] = { command_list };

    // TODO batch command lists instead of waiting
    m_mtx.lock();
    m_command_queue->ExecuteCommandLists(1, ppCommandLists);
    UINT64 handle = m_fence->signal();

    m_command_allocator_queue.emplace(Allocator_queue_item{handle, allocator});
    m_free_command_list_queue.emplace(command_list);
    m_mtx.unlock();

    return handle;
}

// ------------------------------------------------------------------------------------------------

void Command_queue::flush()
{
    m_fence->wait(m_fence->signal());
}

// ------------------------------------------------------------------------------------------------

ComPtr<ID3D12CommandAllocator> Command_queue::create_command_allocator()
{
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> command_allocator;
    throw_on_failure(m_app->get_device()->CreateCommandAllocator(
        m_command_list_type, IID_PPV_ARGS(&command_allocator)),
        "Failed to create command allocator.", SRC);
    return command_allocator;
}

// ------------------------------------------------------------------------------------------------

ComPtr<D3DCommandList> Command_queue::create_command_list(ID3D12CommandAllocator* allocator)
{
    ComPtr<D3DCommandList> command_list;
    throw_on_failure(m_app->get_device()->CreateCommandList(
        0, m_command_list_type, allocator, nullptr, IID_PPV_ARGS(&command_list)),
        "Failed to create command list.", SRC);

    return command_list;
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

Fence::Fence(Base_application* app, Command_queue* queue)
    : m_app(app)
    , m_mtx()
    , m_command_queue(queue)
{
    throw_on_failure(
        m_app->get_device()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)),
        "Failed to create fence.", SRC);

    m_fence_value = 1;
    m_fence_event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (!m_fence_event)
        log_error("Failed to create fence event.", SRC);
}

// ------------------------------------------------------------------------------------------------

Fence::~Fence()
{
    wait(m_fence_value - 1);
    CloseHandle(m_fence_event);
}

// ------------------------------------------------------------------------------------------------

UINT64 Fence::signal()
{
    m_mtx.lock();
    UINT64 handle = m_fence_value++;
    throw_on_failure(m_command_queue->get_queue()->Signal(m_fence.Get(), handle),
        "Failed to signal fence.", SRC);

    m_mtx.unlock();
    return handle;
}

// ------------------------------------------------------------------------------------------------

bool Fence::is_completed(const UINT64& handle) const
{
    return m_fence->GetCompletedValue() >= handle;
}

// ------------------------------------------------------------------------------------------------

bool Fence::wait(const UINT64& handle) const
{
    if (m_fence->GetCompletedValue() < handle)
    {
        throw_on_failure(m_fence->SetEventOnCompletion(handle, m_fence_event),
            "Failed to wait on fence", SRC);

        WaitForSingleObjectEx(m_fence_event, INFINITE, FALSE);
        return m_fence->GetCompletedValue() >= handle;
    }
    return true;
}

}}} // mi::examples::mdl_d3d12
