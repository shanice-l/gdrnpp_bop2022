//clang++ -o build/query_device cpp/query_devices.cpp -Iegl_include -lpthread -ldl -lGL -lEGL -std=c++11
// https://github.com/fxia22/egl_example/blob/master/query_devices.cpp
#include <stdio.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GL/gl.h>

struct EGLInternalData2 {
    bool m_isInitialized;

    int m_windowWidth;
    int m_windowHeight;
    int m_renderDevice;

    EGLBoolean success;
    EGLint num_configs;
    EGLConfig egl_config;
    EGLSurface egl_surface;
    EGLContext egl_context;
    EGLDisplay egl_display;

    EGLInternalData2()
    : m_isInitialized(false),
    m_windowWidth(0),
    m_windowHeight(0) {}
};


int main(){
    // Load extensions
    // https://github.com/pmh47/dirt/blob/master/csrc/gl_common.h#L42
    auto const eglQueryDevicesEXT = (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
    auto const eglQueryDeviceAttribEXT = (PFNEGLQUERYDEVICEATTRIBEXTPROC) eglGetProcAddress("eglQueryDeviceAttribEXT");
    auto const eglGetPlatformDisplayEXT = (PFNEGLGETPLATFORMDISPLAYEXTPROC) eglGetProcAddress("eglGetPlatformDisplayEXT");
    if (!eglQueryDevicesEXT || !eglQueryDeviceAttribEXT || !eglGetPlatformDisplayEXT) {
        fprintf(stderr, "extensions: eglQueryDevicesEXT, eglQueryDeviceAttribEXT and eglGetPlatformDisplayEXT not available.\n");
        exit(EXIT_FAILURE);
    }
    printf("query devices:\n");

    int m_windowWidth;
    int m_windowHeight;
    int m_renderDevice;

    EGLBoolean success;
    EGLint num_configs;
    EGLConfig egl_config;
    EGLSurface egl_surface;
    EGLContext egl_context;
    EGLDisplay egl_display;

    m_windowWidth = 256;
    m_windowHeight = 256;
    m_renderDevice = -1;

    EGLint egl_config_attribs[] = {
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_DEPTH_SIZE, 8,
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_NONE};

    EGLint egl_pbuffer_attribs[] = {
        EGL_WIDTH, m_windowWidth,
        EGL_HEIGHT, m_windowHeight,
        EGL_NONE,
    };

    EGLInternalData2* m_data = new EGLInternalData2();

    // Query EGL Devices
    const int max_devices = 32;
    EGLDeviceEXT egl_devices[max_devices];
    EGLint num_devices = 0;
    EGLint egl_error = eglGetError();
    if (!eglQueryDevicesEXT(max_devices, egl_devices, &num_devices) ||
        egl_error != EGL_SUCCESS) {
        printf("eglQueryDevicesEXT Failed.\n");
        m_data->egl_display = EGL_NO_DISPLAY;
    }

    printf("num devices: %d\n", num_devices);

    return 0;
}
