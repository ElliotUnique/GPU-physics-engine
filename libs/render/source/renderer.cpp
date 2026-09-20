
#ifndef ENGINE_D3D_DEBUG
#define ENGINE_D3D_DEBUG 0
#endif
#define WIN_API 1
// Standard libs
#include <string>
// My libs
#include "common/umbrella.h"
#include "cuda-engine/cuda-engine.h"
#include "render/renderer.h"

[[noreturn]] static void throwCudaFailure(
    const char* what,
    int rc
)
{
    std::string msg = std::string(what) + " failed (rc=" + std::to_string(rc) + ")";
    if (rc == -1) {
        const int e  = cuda_engine_last_error();
        msg         += ": ";
        msg         += cuda_engine_error_name(e);
    } else if (rc == -2) {
        msg += ": CUDA device does not match D3D12 adapter LUID";
    }
    throw std::runtime_error(msg);
}
namespace rend
{
// ===== Method Definitions ================================
Renderer::Renderer(
    HWND hWnd_,
    size_t rect_w,
    size_t rect_h
)
{
    rect_width  = rect_w;
    rect_height = rect_h;
    hWnd        = hWnd_;
    createDeviceResources();
}

Renderer::~Renderer()
{
    if (queue && d3dFence && hFenceEvent) { flush(); }
    destroySizeResources();
    cuda_engine_shutdown();
    if (hFenceEvent) { CloseHandle(hFenceEvent); }
    if (hFrameLatency) { CloseHandle(hFrameLatency); }
}

void Renderer::createDeviceResources()
{
    // D3D12 device and queue init
    UINT factoryFlags = 0;
#ifdef ENGINE_D3D_DEBUG
    {
        ComPtr<ID3D12Debug> debug;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) {
            debug->EnableDebugLayer();
            factoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
        }
    }
#endif

    ComPtr<IDXGIFactory6> factory;
    ThrowIfFailed(CreateDXGIFactory2(factoryFlags, IID_PPV_ARGS(&factory)));

    ComPtr<IDXGIAdapter1> adapter;

    for (UINT i = 0; factory->EnumAdapterByGpuPreference(
                         i,
                         DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                         IID_PPV_ARGS(&adapter)
                     ) != DXGI_ERROR_NOT_FOUND;
         ++i) {
        DXGI_ADAPTER_DESC1 desc{};
        adapter->GetDesc1(&desc);

        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
            adapter.Reset();
            continue;
        }

        // Probe with a null out-param: tests capability without creating anything.
        if (SUCCEEDED(
                D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, _uuidof(ID3D12Device), nullptr)
            )) {
            adapterLuid = desc.AdapterLuid;
            break;
        }
        adapter.Reset();
    }
    if (!adapter) { throw std::runtime_error("No suitable D3D12 adapter"); }

    ThrowIfFailed(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)));

    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type     = D3D12_COMMAND_LIST_TYPE_DIRECT;
    qd.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
    qd.Flags    = D3D12_COMMAND_QUEUE_FLAG_NONE;
    qd.NodeMask = 0;

    ThrowIfFailed(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)));

#ifdef ENGINE_D3D_DEBUG
    {
        ComPtr<ID3D12InfoQueue> infoQueue;
        if (SUCCEEDED(device.As(&infoQueue))) {
            infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, TRUE);
            infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, TRUE);
        }
    }
#endif

    BOOL tearing = FALSE;
    ComPtr<IDXGIFactory5> factory5;
    if (SUCCEEDED(factory.As(&factory5))) {
        if (FAILED(
                factory5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &tearing, sizeof(tearing))
            )) {
            tearing = FALSE;
        }
    }
    tearingSupported = (tearing == TRUE);

    // Swap-chain init
    DXGI_SWAP_CHAIN_DESC1 swapchain_desc{};
    swapchain_desc.Width       = (uint32_t)rect_width;
    swapchain_desc.Height      = (uint32_t)rect_height;
    swapchain_desc.Format      = DXGI_FORMAT_R8G8B8A8_UNORM;
    swapchain_desc.Stereo      = FALSE;
    swapchain_desc.SampleDesc  = {1, 0};
    swapchain_desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapchain_desc.BufferCount = kFrameCount;
    swapchain_desc.Scaling     = DXGI_SCALING_STRETCH; // DXGI_SCALING_NONE;
    swapchain_desc.SwapEffect  = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    swapchain_desc.AlphaMode   = DXGI_ALPHA_MODE_IGNORE;
    swapchain_desc.Flags       = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
    if (tearingSupported) { swapchain_desc.Flags |= DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING; }

    ComPtr<IDXGISwapChain1> swapchain1;
    ThrowIfFailed(factory->CreateSwapChainForHwnd(
        queue.Get(), // note: the QUEUE, not the device
        hWnd,
        &swapchain_desc,
        nullptr, // no fullscreen desc — borderless-fullscreen on your terms later
        nullptr, // no output restriction
        &swapchain1
    ));

    ThrowIfFailed(swapchain1.As(&swapChain));

    ThrowIfFailed(factory->MakeWindowAssociation(hWnd, DXGI_MWA_NO_ALT_ENTER));

    hFrameLatency = swapChain->GetFrameLatencyWaitableObject();
    if (!hFrameLatency) { ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError())); }
    swapChain->SetMaximumFrameLatency(2);

    D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc{};
    rtvHeapDesc.NumDescriptors = kFrameCount;
    rtvHeapDesc.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtvHeapDesc.Flags          = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

    ThrowIfFailed(device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&rtvHeap)));

    rtvStride = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    for (UINT i = 0; i < kFrameCount; ++i) {
        ThrowIfFailed(
            device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocators[i]))
        );
    }

    ThrowIfFailed(device->CreateCommandList(
        0,
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        allocators[0].Get(),
        nullptr,
        IID_PPV_ARGS(&cmdList)
    ));
    ThrowIfFailed(cmdList->Close()); // lists are born open; close so Reset works

    ThrowIfFailed(device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&cudaFence)));
    ThrowIfFailed(device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&d3dFence)));

    hFenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!hFenceEvent) { ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError())); }

    HANDLE hCudaFence = nullptr, hD3dFence = nullptr;
    ThrowIfFailed(device->CreateSharedHandle(cudaFence.Get(), nullptr, GENERIC_ALL, nullptr, &hCudaFence));
    ThrowIfFailed(device->CreateSharedHandle(d3dFence.Get(), nullptr, GENERIC_ALL, nullptr, &hD3dFence));

    CudaEngineDeviceDesc dd{};
    dd.cuda_fence_handle = hCudaFence;
    dd.d3d_fence_handle  = hD3dFence;
    memcpy(dd.adapter_luid, &adapterLuid, 8);

    const int rc = cuda_engine_init(&dd);
    CloseHandle(hCudaFence);
    CloseHandle(hD3dFence);
    if (rc != 0) { throwCudaFailure("cuda_engine_init", rc); }

    createSizeResources();
}

void Renderer::createSizeResources()
{
    // --- backbuffer references + RTVs ---
    D3D12_CPU_DESCRIPTOR_HANDLE rtvBase = rtvHeap->GetCPUDescriptorHandleForHeapStart();
    for (UINT i = 0; i < kFrameCount; ++i) {
        ThrowIfFailed(swapChain->GetBuffer(i, IID_PPV_ARGS(&backBuffers[i])));
        D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle  = rtvBase;
        rtvHandle.ptr                         += static_cast<SIZE_T>(i) * rtvStride;
        device->CreateRenderTargetView(backBuffers[i].Get(), nullptr, rtvHandle);
    }
    frameIndex = swapChain->GetCurrentBackBufferIndex();

    // Footprint
    D3D12_RESOURCE_DESC texDesc{};
    texDesc.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texDesc.Alignment        = 0;
    texDesc.Width            = (uint32_t)rect_width;
    texDesc.Height           = (uint32_t)rect_height;
    texDesc.DepthOrArraySize = 1;
    texDesc.MipLevels        = 1;
    texDesc.Format           = DXGI_FORMAT_R8G8B8A8_UNORM;
    texDesc.SampleDesc       = {1, 0};
    texDesc.Layout           = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    texDesc.Flags            = D3D12_RESOURCE_FLAG_NONE;

    UINT numRows        = 0;
    UINT64 rowSizeBytes = 0;
    UINT64 totalBytes   = 0;

    device->GetCopyableFootprints(&texDesc, 0, 1, 0, &footprint, &numRows, &rowSizeBytes, &totalBytes);
    buf_row_pitch = footprint.Footprint.RowPitch;
    buf_bytes     = static_cast<UINT64>(buf_row_pitch) * rect_height;

    // Shared buffers
    D3D12_HEAP_PROPERTIES defaultHeap{};
    defaultHeap.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC bufDesc{};
    bufDesc.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufDesc.Alignment        = 0;
    bufDesc.Width            = buf_bytes;
    bufDesc.Height           = 1;
    bufDesc.DepthOrArraySize = 1;
    bufDesc.MipLevels        = 1;
    bufDesc.Format           = DXGI_FORMAT_UNKNOWN;
    bufDesc.SampleDesc       = {1, 0};
    bufDesc.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    bufDesc.Flags            = D3D12_RESOURCE_FLAG_NONE;

    for (UINT i = 0; i < kFrameCount; ++i) {
        ThrowIfFailed(device->CreateCommittedResource(
            &defaultHeap,
            D3D12_HEAP_FLAG_SHARED,
            &bufDesc,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&sharedBuffers[i])
        ));
    }

    HANDLE hShared[kFrameCount] = {};
    for (UINT i = 0; i < kFrameCount; ++i) {
        ThrowIfFailed(
            device->CreateSharedHandle(sharedBuffers[i].Get(), nullptr, GENERIC_ALL, nullptr, &hShared[i])
        );
    }

    const D3D12_RESOURCE_ALLOCATION_INFO ai = device->GetResourceAllocationInfo(0, 1, &bufDesc);

    CudaEngineBufferDesc bufd{};
    for (UINT i = 0; i < kFrameCount; ++i) { bufd.resource_handles[i] = hShared[i]; }
    bufd.buffer_count     = kFrameCount;
    bufd.allocation_bytes = ai.SizeInBytes;
    bufd.buf_bytes        = buf_bytes;
    bufd.buf_row_pitch    = buf_row_pitch;
    bufd.width            = (uint32_t)rect_width;
    bufd.height           = (uint32_t)rect_height;

    const int rc = cuda_engine_bind_buffers(&bufd);
    for (UINT i = 0; i < kFrameCount; ++i) { CloseHandle(hShared[i]); }
    if (rc != 0) { throwCudaFailure("cuda_engine_bind_buffers", rc); }
}

void Renderer::destroySizeResources()
{
    cuda_engine_unbind_buffers(); // syncs the stream, then frees mappings

    for (UINT i = 0; i < kFrameCount; ++i) {
        sharedBuffers[i].Reset();
        backBuffers[i].Reset(); // MUST be released before ResizeBuffers
    }
}

void Renderer::resize(
    UINT rect_w,
    UINT rect_h
)
{
    do {
        (rect_w == rect_h);
    } while (0);

    throw std::logic_error("Renderer::resize not implemented");

    // The sequence, when you come back to this:
    //   if (w == 0 || h == 0) return;              // minimised
    //   flush();                                    // GPU must be idle
    //   destroySizeResources();
    //   ThrowIfFailed(swapChain->ResizeBuffers(
    //       kFrameCount, w, h, DXGI_FORMAT_UNKNOWN,
    //       DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT));
    //   createSizeResources(w, h);
    //   frameCounter continues; fence timeline is unaffected.
}

void Renderer::renderFrame(OutputBuffer& buff)
{
    // 1. Recycle: has the GPU finished the last frame that used this allocator?
    if (d3dFence->GetCompletedValue() < fenceValues[frameIndex]) {
        ThrowIfFailed(d3dFence->SetEventOnCompletion(fenceValues[frameIndex], hFenceEvent));
        WaitForSingleObject(hFenceEvent, INFINITE);
    }

    ThrowIfFailed(allocators[frameIndex]->Reset());
    ThrowIfFailed(cmdList->Reset(allocators[frameIndex].Get(), nullptr));

    const uint32_t bufIndex  = static_cast<uint32_t>(frameCounter & 1);
    const uint64_t waitValue = (frameCounter >= 2) ? (frameCounter - 1) : 0;

    const View tick_output = buff.readTarget();
    const int rc = cuda_engine_render_frame(tick_output, waitValue, frameCounter + 1, frameCounter, bufIndex);

    if (rc != 0) { throwCudaFailure("cuda_engine_render_frame", rc); }

    // 2. PRESENT -> RENDER_TARGET
    D3D12_RESOURCE_BARRIER barrier{};

    // backbuffer: PRESENT -> COPY_DEST
    barrier.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags                  = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource   = backBuffers[frameIndex].Get();
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_DEST;

    cmdList->ResourceBarrier(1, &barrier);

    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource        = backBuffers[frameIndex].Get();
    dst.Type             = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.SubresourceIndex = 0;

    D3D12_TEXTURE_COPY_LOCATION src{};
    src.pResource       = sharedBuffers[bufIndex].Get();
    src.Type            = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint = footprint;

    cmdList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_PRESENT;
    cmdList->ResourceBarrier(1, &barrier);

    ThrowIfFailed(cmdList->Close());
    ThrowIfFailed(queue->Wait(cudaFence.Get(), frameCounter + 1));
    ID3D12CommandList* lists[] = {cmdList.Get()};

    queue->ExecuteCommandLists(1, lists);
    const UINT sync  = vsync ? 1u : 0u;
    const UINT flags = (!vsync && tearingSupported) ? DXGI_PRESENT_ALLOW_TEARING : 0u;
    checkRuntime(swapChain->Present(sync, flags));

    checkRuntime(queue->Signal(d3dFence.Get(), frameCounter + 1));
    fenceValues[frameIndex] = frameCounter + 1;
    ++frameCounter;

    frameIndex = swapChain->GetCurrentBackBufferIndex();
}

void Renderer::flush()
{
    const UINT64 v = frameCounter + 1;
    ++frameCounter;
    checkRuntime(queue->Signal(d3dFence.Get(), v));
    if (d3dFence->GetCompletedValue() < v) {
        ThrowIfFailed(d3dFence->SetEventOnCompletion(v, hFenceEvent));
        WaitForSingleObject(hFenceEvent, INFINITE);
    }
}
} // namespace rend