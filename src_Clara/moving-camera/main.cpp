// main.cpp
// Boilerplate for interactive CUDA ray tracer using Freeglut + GLEW + CUDA/GL interop.
// Compile with:
//   nvcc -o raytracer main.cpp raytracer.cu $(pkg-config --cflags --libs glew glut) -lcuda

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cstdio>
#include <cstdlib>

#include "raytracer.cuh"
#include "bvh.h"
#include "camera.h"
#include "hittable.h"
#include "vec3.h"


// ---------------------------------------------------------------------------
// Window / render state
// ---------------------------------------------------------------------------
float ASP_R = 16.0f/9.0f;
static int WIN_W = 1200;
static int WIN_H = int(float(WIN_W)/ASP_R);

static int num_objects = 488;
int num_pixels = WIN_W * WIN_H;        
int samples_per_pixel = 5;             // Sampels per pixel when camera is moving
static bool camera_moving = false;
static int interactive_depth = 2;
static int quality_depth = 8;


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

// ---------------------------------------------------------------------------
// Set all inital values for camera and create all pointers
// ---------------------------------------------------------------------------
// World and list of objects in it
static hittable** d_list;
static hittable** d_world;

// Host and Device Camera
float vfov = 20.0f;
point3 look_from = point3(13.0f, 2.0f, 3.0f);
point3 look_at   = point3(0.0f, 0.0f, 0.0f);
vec3 vup = vec3(0.0f, 1.0f, 0.0f);
float defocus_angle = 0.0f;
float focus_dist    = 10.0f;
static float cam_speed = 0.5f;                  // Camera movement speed.
static float turn_speed = 0.5f;                 // Turning speed
static camera h_cam;                
static camera* d_cam = nullptr;   

// BVH arrays
static bvh_array* d_bvh = nullptr;
static int* d_bvh_size = nullptr;

// Needed for random number generation
static curandState* d_states = nullptr;
static unsigned long seed = 9999999;

// Accumulation buffer
static float3* d_accum = nullptr;

static float yaw   = -90.0f;  // degrees, -90 faces along -Z initially
static float pitch =   0.0f;  // degrees


// ---------------------------------------------------------------------------
// Initialize all buffers only needed for rendering (Updated each frame)
// ---------------------------------------------------------------------------
static void initRenderBuffers(){
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
// Initialize all buffers (Should only be called in the beginning)
// ---------------------------------------------------------------------------
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
    CUDA_CHECK(cudaMalloc(&d_list, num_objects * sizeof(hittable*)));
    CUDA_CHECK(cudaMalloc(&d_world, sizeof(hittable*)));
    CUDA_CHECK(cudaMalloc(&d_bvh, 4*num_objects*sizeof(bvh_array)));
    CUDA_CHECK(cudaMalloc(&d_bvh_size, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_states, num_pixels * sizeof(curandState)));
    CUDA_CHECK(cudaMalloc(&d_cam, sizeof(camera)));
    CUDA_CHECK(cudaMalloc(&d_accum, WIN_W * WIN_H * sizeof(float3)));

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
// Update Camera
// ---------------------------------------------------------------------------
static void updateCamera(camera* d_cam){
    // Forward vector (normalised direction toward target)
    vec3 forward = unit_vector(look_at - look_from);
    vec3 right   = unit_vector(cross(forward, vec3(0.0f, 1.0f, 0.0f)));

    h_cam.look_from = look_from;
    h_cam.look_at   = look_at;
    h_cam.move_to_device(d_cam);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaMemset(d_accum, 0, WIN_W * WIN_H * sizeof(float3)));
    frameCount = 0;  // reset sample counter
}

// Rotation computations
static void recomputeLookAt()
{
    // Convert yaw/pitch to a direction vector
    float y = degrees_to_radians(yaw);
    float p = degrees_to_radians(pitch);

    vec3 forward(
        cosf(p) * cosf(y),
        sinf(p),
        cosf(p) * sinf(y)
    );
    forward = unit_vector(forward);
    look_at = look_from + forward;
}


// ---------------------------------------------------------------------------
// Freeglut callbacks
// ---------------------------------------------------------------------------

// Called every frame
static void display()
{

    // Sanity check pointers before launch
    if (!d_cam || !d_bvh || !d_list || !d_states || !cudaFramebuffer) {
        fprintf(stderr, "Null pointer before kernel launch!\n");
        return;
    }
    int max_depth = camera_moving ? interactive_depth : quality_depth;
    CUDA_CHECK(cudaMemcpyAsync(d_cam, &h_cam, sizeof(camera),
                               cudaMemcpyHostToDevice, renderStream));
    // Launch kernel — async, returns immediately
    launchAccumRayTracer(d_bvh, d_list, d_accum, cudaFramebuffer, WIN_W, WIN_H,
                  max_depth, samples_per_pixel, frameCount+1, d_cam, d_states, renderStream);

    // Async copy device -> pinned host, queued behind the kernel on same stream
    CUDA_CHECK(cudaMemcpyAsync(hostFramebuffer, cudaFramebuffer,
                               WIN_W * WIN_H * sizeof(uchar4),
                               cudaMemcpyDeviceToHost, renderStream));
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
    initRenderBuffers();  // Recreate PBO / texture at new size
}

static void specialKeyboard(int key, int /*x*/, int /*y*/)
{
    vec3 forward = unit_vector(look_at - look_from);
    vec3 right   = unit_vector(cross(forward, vec3(0.0f, 1.0f, 0.0f)));

    switch (key) {
        case GLUT_KEY_UP:    pitch += turn_speed;  break;
        case GLUT_KEY_DOWN:  pitch -= turn_speed;  break;
        case GLUT_KEY_LEFT:  yaw   -= turn_speed;  break;
        case GLUT_KEY_RIGHT: yaw   += turn_speed;  break;
    }

    // Clamp pitch to avoid gimbal flip at 90 degrees
    if (pitch >  89.0f) pitch =  89.0f;
    if (pitch < -89.0f) pitch = -89.0f;

    recomputeLookAt();
    updateCamera(d_cam);
    glutPostRedisplay();
}
// Keyboard: press Escape or Q to quit
static void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    vec3 forward = unit_vector(look_at - look_from);
    vec3 right   = unit_vector(cross(forward, vec3(0.0f, 1.0f, 0.0f)));
    
    switch (key) {
        case 'w': camera_moving = true;
                  look_at += cam_speed * forward;
                  look_from += cam_speed * forward;  break;
        case 's': camera_moving = true;
                  look_at -= cam_speed * forward;
                  look_from -= cam_speed * forward;  break;
        case 'a': camera_moving = true;
                  look_at -= cam_speed * right;
                  look_from -= cam_speed * right;    break;
        case 'd': camera_moving = true;
                  look_at += cam_speed * right;
                  look_from += cam_speed * right;    break;
        case 'e': camera_moving = true;
                  look_at.e[1] += cam_speed;
                  look_from.e[1] += cam_speed;       break;  // up
        case 'q': camera_moving = true;
                  look_at.e[1] -= cam_speed;
                  look_from.e[1] -= cam_speed;       break;  // down
        case 'x':                                            // Escape
        if (cudaFramebuffer) {
            cudaFree(cudaFramebuffer);
            cudaFreeHost(hostFramebuffer);
            cudaStreamDestroy(renderStream);
            glDeleteTextures(1, &displayTex);
        }
        glutLeaveMainLoop();
        break;
        case 27:   // Escape

        default:
            break;
    }
    updateCamera(d_cam);
    glutPostRedisplay();
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

    // --- Set up CUDA device ---
    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("CUDA device     : %s\n", prop.name);


    // --- Create PBO + texture ---
    initBuffers();

    // Set up random numbers, the world and build BVH
    printf("d_states ptr: %p\n", (void*)d_states);
    printf("WIN_W=%d WIN_H=%d num_pixels=%d\n", WIN_W, WIN_H, num_pixels);
    printf("Allocated size: %zu bytes\n", WIN_W * WIN_H * sizeof(curandState));
    initRand(d_states, WIN_W, WIN_H, seed);
    CUDA_CHECK(cudaGetLastError());  // catch any kernel errors from initRand
    initWorld(d_bvh, d_list, d_world, d_bvh_size, num_objects, WIN_W, WIN_H);
    CUDA_CHECK(cudaGetLastError());  // catch any kernel errors from initWorld
    initCamera(h_cam, d_cam, WIN_H, WIN_W, samples_per_pixel, quality_depth, vfov, look_from, look_at, vup, defocus_angle, focus_dist);
    printf("d_cam ptr: %p\n", (void*)d_cam);
    printf("d_bvh ptr: %p\n", (void*)d_bvh);
    printf("d_list ptr: %p\n", (void*)d_list);
    printf("d_states ptr: %p\n", (void*)d_states);

    // --- Register callbacks ---
    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutSpecialFunc(specialKeyboard);

    // --- Enter render loop ---
    glutMainLoop();

    return EXIT_SUCCESS;
}
