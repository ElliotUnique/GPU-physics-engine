#pragma once
#include <windows.h>
#include <windowsx.h>

#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <stdexcept>
#include <stdint.h>
#include <thread>
#include <utility>

#include "common/umbrella.h"
#include "cuda-interface.h"

namespace rend
{
using Microsoft::WRL::ComPtr;

struct DeviceRemovedException : std::runtime_error
{
    HRESULT reason;
    explicit DeviceRemovedException(HRESULT r)
        : std::runtime_error("D3D12 device removed"),
          reason(r)
    {
    }
};

inline void ThrowIfFailed(HRESULT hr)
{
    if (FAILED(hr)) {
        __debugbreak();
        throw std::runtime_error("D3D12 call failed");
    }
}

constexpr UINT kFrameCount = 2;

class Renderer
{
  public:
    void setVSync(bool on) { vsync = on; }
    HANDLE frameLatencyWaitable() const { return hFrameLatency; }

    void renderFrame(OutputBuffer& buff);
    void resize(
        UINT w,
        UINT h
    );

    Renderer(
        HWND hWnd,
        size_t rect_w,
        size_t rect_h
    );
    ~Renderer();

  private:
    // D3D attributes
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<IDXGISwapChain3> swapChain;
    ComPtr<ID3D12DescriptorHeap> rtvHeap;
    ComPtr<ID3D12Resource> backBuffers[kFrameCount];
    ComPtr<ID3D12CommandAllocator> allocators[kFrameCount];
    ComPtr<ID3D12GraphicsCommandList> cmdList;
    ComPtr<ID3D12Resource> sharedBuffers[kFrameCount];
    ComPtr<ID3D12Fence> cudaFence; // signalled by CUDA only:  "kernel N done"
    ComPtr<ID3D12Fence> d3dFence;  // signalled by D3D12 only: "copy N done"

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};

    // Win 32 attributes
    HWND hWnd                       = nullptr;
    HANDLE hFenceEvent              = nullptr;
    HANDLE hFrameLatency            = nullptr;
    UINT rtvStride                  = 0;
    UINT frameIndex                 = 0;
    uint64_t frameCounter           = 0;
    UINT64 fenceValues[kFrameCount] = {};
    LUID adapterLuid{};
    UINT buf_row_pitch    = 0;
    UINT64 buf_bytes      = 0;
    BOOL tearingSupported = false;
    BOOL vsync            = true;
    UINT swapChainFlags   = 0;

    // Other attributes
    size_t rect_width  = 0;
    size_t rect_height = 0;
    CameraSpan cam{};

    void createDeviceResources();
    void destroySizeResources();
    void createSizeResources();
    void flush();
    void checkRuntime(HRESULT hr) const
    {
        if (hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_RESET) {
            throw DeviceRemovedException(device->GetDeviceRemovedReason());
        }
        ThrowIfFailed(hr);
    }

    Renderer(const Renderer&)            = delete;
    Renderer& operator=(const Renderer&) = delete;
};

void winEngineGUI(
    GUISettings& settings,
    OutputBuffer& buff,
    std::atomic<Status>& gui_status,
    Bell& bell
);
LRESULT CALLBACK myWindowProc(
    HWND hWnd,
    UINT msg,
    WPARAM wp,
    LPARAM lp
);

} // namespace rend