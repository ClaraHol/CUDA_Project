// main.cpp
// Boilerplate for interactive CUDA ray tracer using Freeglut + GLEW + CUDA/GL interop.
// Compile with:
//   nvcc -o raytracer main.cpp raytracer.cu $(pkg-config --cflags --libs glew glut) -lcuda

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cstdio>
#include <cstdlib>
#include "raytracer.cuh"

// ---------------------------------------------------------------------------
// Window / render state
// ---------------------------------------------------------------------------
static int WIN_W = 1280;
static int WIN_H = 720;

// PBO (pixel buffer object) — CUDA writes into this, OpenGL reads from it
static GLuint pbo = 0;
static cudaGraphicsResource* cudaPboResource = nullptr;

// Texture we blit to the screen
static GLuint displayTex = 0;

// Simple frame counter for diagnostics
static int frameCount = 0;

// ---------------------------------------------------------------------------
// CUDA error helper
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Create / recreate PBO and texture at the current window size
// ---------------------------------------------------------------------------

static uchar4* cudaFramebuffer = nullptr;
static uchar4* hostFramebuffer = nullptr;
static cudaStream_t renderStream;

static void initBuffers()
{
    if (cudaFramebuffer) {
        cudaFree(cudaFramebuffer);
        cudaFreeHost(hostFramebuffer);
        glDeleteTextures(1, &displayTex);
        cudaStreamDestroy(renderStream);
    }

    CUDA_CHECK(cudaMalloc(&cudaFramebuffer,
                          WIN_W * WIN_H * sizeof(uchar4)));
    CUDA_CHECK(cudaMallocHost(&hostFramebuffer,
                              WIN_W * WIN_H * sizeof(uchar4)));
    CUDA_CHECK(cudaStreamCreate(&renderStream));

    glGenTextures(1, &displayTex);
    glBindTexture(GL_TEXTURE_2D, displayTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,
                 WIN_W, WIN_H, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);
}

// ---------------------------------------------------------------------------
// Freeglut callbacks
// ---------------------------------------------------------------------------

// Called every frame
static void display()
{
    //printf("cudaFramebuffer ptr: %p\n", (void*)cudaFramebuffer);
    //printf("WIN_W=%d WIN_H=%d\n", WIN_W, WIN_H);
    // Launch kernel — async, returns immediately
    launchRayTracer(cudaFramebuffer, WIN_W, WIN_H, frameCount, renderStream);

    // Async copy device -> pinned host, queued behind the kernel on same stream
    CUDA_CHECK(cudaMemcpyAsync(hostFramebuffer, cudaFramebuffer,
                               WIN_W * WIN_H * sizeof(uchar4),
                               cudaMemcpyDeviceToHost, renderStream));
    CUDA_CHECK(cudaStreamSynchronize(renderStream));

    // Verify some pixels are non-zero
    uchar4 sample = hostFramebuffer[WIN_W * WIN_H / 2];
    //printf("Centre pixel: %d %d %d\n", sample.x, sample.y, sample.z);

    // Wait for both kernel and copy to finish
    CUDA_CHECK(cudaStreamSynchronize(renderStream));

    // Upload to GL texture and display
    glBindTexture(GL_TEXTURE_2D, displayTex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIN_W, WIN_H,
                    GL_RGBA, GL_UNSIGNED_BYTE, hostFramebuffer);

    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, displayTex);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glBegin(GL_QUADS);
        glTexCoord2f(0, 0);  glVertex2f(0, 0);
        glTexCoord2f(1, 0);  glVertex2f(1, 0);
        glTexCoord2f(1, 1);  glVertex2f(1, 1);
        glTexCoord2f(0, 1);  glVertex2f(0, 1);
    glEnd();

    glDisable(GL_TEXTURE_2D);
    glutSwapBuffers();
    ++frameCount;
}
// Keep rendering as fast as possible
static void idle()
{
    glutPostRedisplay();
}

// Handle window resize
static void reshape(int w, int h)
{
    if (w == 0 || h == 0) return;
    WIN_W = w;
    WIN_H = h;
    glViewport(0, 0, w, h);
    initBuffers();  // Recreate PBO / texture at new size
}

// Keyboard: press Escape or Q to quit
static void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key) {
        case 27:   // Escape
        case 'q':
        if (cudaFramebuffer) {
            cudaFree(cudaFramebuffer);
            cudaFreeHost(hostFramebuffer);
            cudaStreamDestroy(renderStream);
            glDeleteTextures(1, &displayTex);
        }
        glutLeaveMainLoop();
        break;

        // Add your own keys here, e.g. to move the camera
        // case 'w': camera.moveForward(); break;

        default:
            break;
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // --- Freeglut init ---
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(WIN_W, WIN_H);
    glutCreateWindow("CUDA Ray Tracer");

    // --- GLEW init (must be after glutCreateWindow) ---
    GLenum glewErr = glewInit();
    if (glewErr != GLEW_OK) {
        fprintf(stderr, "GLEW init failed: %s\n", glewGetErrorString(glewErr));
        return EXIT_FAILURE;
    }

    // --- Print basic info ---
    printf("OpenGL renderer : %s\n", glGetString(GL_RENDERER));
    printf("OpenGL version  : %s\n", glGetString(GL_VERSION));

    // Check for the PBO extension (should always be present on any modern GPU)
    if (!GLEW_ARB_pixel_buffer_object) {
        fprintf(stderr, "GL_ARB_pixel_buffer_object not supported!\n");
        return EXIT_FAILURE;
    }
    glClear(GL_COLOR_BUFFER_BIT);
    glutSwapBuffers();
    glFinish();  // Force GL to fully initialize before CUDA attaches

    // --- Set up CUDA device ---
    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("CUDA device     : %s\n", prop.name);


    // --- Create PBO + texture ---
    glClear(GL_COLOR_BUFFER_BIT);
    glutSwapBuffers();
    glutSwapBuffers();  // Two swaps to cycle through both front and back buffers
    glFinish();

    initBuffers();

    // --- Register callbacks ---
    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);

    // --- Enter render loop ---
    glutMainLoop();

    return EXIT_SUCCESS;
}
