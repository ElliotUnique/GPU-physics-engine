#define WIN_API 1
// Windows libs
#include <windows.h>
#include <windowsx.h>
// D3D12/DX libs
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
using Microsoft::WRL::ComPtr;
// Standard libs
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <stdlib.h>
// My libs
#include "common/umbrella.h"
#include "cuda-engine/cuda-engine.h"
#include "render/renderer.h"

namespace rend
{

static void reportError(const char* msg)
{
    OutputDebugStringA(msg);
    OutputDebugStringA("\n");
    std::fprintf(stderr, "%s\n", msg);
}

// Global variables
int32_t wcorner_x = 0;
int32_t wcorner_y = 22;

void winEngineGUI(
    GUISettings& settings,
    OutputBuffer& buff,
    std::atomic<Status>& my_status,
    Bell& bell
)
try {
    std::mutex m;
    std::condition_variable cv;

    HINSTANCE hInstance = GetModuleHandleW(nullptr);
    // Register class, create window, initialise frame timer, run simulation.
    WNDCLASSEXW wc{0};
    wc.cbSize        = sizeof(wc);
    wc.hbrBackground = nullptr;
    wc.hCursor       = LoadCursor(NULL, IDC_ARROW);
    wc.hInstance     = hInstance;
    wc.lpszClassName = L"MainWindow";
    wc.lpfnWndProc   = myWindowProc;

    if (!RegisterClassExW(&wc)) {
        GetLastError();
        return;
    }

    size_t rect_width  = settings.disp_width;
    size_t rect_height = settings.disp_height;

    RECT rc =
        {(LONG)wcorner_x, (LONG)wcorner_y, (LONG)(wcorner_x + rect_width), (LONG)(wcorner_y + rect_height)};
    UINT dpi = GetDpiForSystem();
    AdjustWindowRectExForDpi(&rc, WS_OVERLAPPEDWINDOW, FALSE, 0, dpi);

    HWND hMain = CreateWindowExW(
        0,
        L"MainWindow",
        L"Main Window",
        WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        rc.left,
        rc.top,
        rc.right - rc.left,
        rc.bottom - rc.top,
        nullptr,
        nullptr,
        hInstance,
        nullptr
    );
    if (!hMain) { return; }

    rend::Renderer rend(hMain, rect_width, rect_height);
    my_status.store(Status::READY);
    my_status.notify_one();

    // Rendering/message loop init
    HANDLE hTimer =
        CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);

    if (!hTimer) {
        [[maybe_unused]] DWORD err = GetLastError();
        MessageBox(hMain, L"CreateWaitableTimer failed", L"Error", MB_OK);
        return;
    }

    auto nowFileTime = []() -> LONGLONG
    {
        FILETIME ft;
        GetSystemTimeAsFileTime(&ft);
        return ((LONGLONG)ft.dwHighDateTime << 32) | ft.dwLowDateTime;
    };

    LONGLONG period_100ns = 10'000'000LL / (settings.FPS > 0 ? settings.FPS : 60);
    LONGLONG deadline     = nowFileTime();

    auto armTimer = [&]()
    {
        deadline           += period_100ns;
        const LONGLONG now  = nowFileTime();
        if (deadline < now) // Resync rather than burn through a backlog
        {
            deadline = now + period_100ns;
        }
        LARGE_INTEGER due;
        due.QuadPart = deadline;                                    // positive => absolute
        SetWaitableTimer(hTimer, &due, 0, nullptr, nullptr, FALSE); // 0 => one-shot
    };

    HANDLE hLatency    = rend.frameLatencyWaitable();
    bool latency_ready = false;
    bool timer_ready   = settings.vsync; // uncapped/vsync: no timer gate
    if (!settings.vsync) { armTimer(); }

    bool running = true;
    while (running) {
        if (my_status.load() == Status::EXIT) { PostMessageW(hMain, WM_CLOSE, 0, 0); }

        if (!latency_ready || !timer_ready) {
            HANDLE waits[2];
            DWORD n             = 0;
            int32_t latency_idx = -1, timer_idx = -1;
            if (!latency_ready) {
                latency_idx = (int32_t)n;
                waits[n++]  = hLatency;
            }
            if (!timer_ready) {
                timer_idx  = (int32_t)n;
                waits[n++] = hTimer;
            }

            const DWORD r = MsgWaitForMultipleObjectsEx(n, waits, INFINITE, QS_ALLINPUT, MWMO_INPUTAVAILABLE);

            if (latency_idx >= 0 && r == WAIT_OBJECT_0 + (DWORD)latency_idx) {
                latency_ready = true;
            } else if (timer_idx >= 0 && r == WAIT_OBJECT_0 + (DWORD)timer_idx) {
                timer_ready = true;
            } else if (r == WAIT_OBJECT_0 + n) {
                MSG msg = {};
                while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
                    if (msg.message == WM_QUIT) { running = false; }
                    TranslateMessage(&msg);
                    DispatchMessageW(&msg);
                }
            }
            continue;
        }

        rend.renderFrame(buff); // waits on cudaFence, copies, presents

        latency_ready = false;
        timer_ready   = settings.vsync;
        if (!settings.vsync) { armTimer(); }
    }
    my_status.store(Status::EXIT);
    bell.ring();
}
catch (const std::exception& e) {
    reportError(e.what());
    my_status.store(Status::EXIT);
    bell.ring();
}

LRESULT CALLBACK rend::myWindowProc(
    HWND hWnd,
    UINT msg,
    WPARAM wp,
    LPARAM lp
)
{
    switch (msg) {
    case WM_CLOSE:
    {
        DestroyWindow(hWnd);
        return 0;
    }
    case WM_DESTROY:
    {
        PostQuitMessage(0);
        return 0;
    }

    case WM_ERASEBKGND:
    {
        return 1; // Suppress GDI background erase
    }
    case WM_SIZE:
    {
        return 0;
    }

    case WM_KEYDOWN:
    {
        if (wp == VK_ESCAPE) {
            DestroyWindow(hWnd);
            return 0;
        }
        break;
    }
    }
    return DefWindowProcW(hWnd, msg, wp, lp);
}
} // namespace rend