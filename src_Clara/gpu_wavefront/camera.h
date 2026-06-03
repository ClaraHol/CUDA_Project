#ifndef CAMERA_H
#define CAMERA_H

#include "hittable.h"
#include "material.h"
#include <omp.h>
#include <vector>

class camera {
    public:
        
        int height = 1200;                  // Image height in pixels
        int width = 400;                    // Image width in pixels
        int samples_per_pixel = 10;         // Count of random samples for each pixel
        int max_depth = 10;                 // Maximum number of recursions in ray_color

        float vfov = 90.0f;                 // Vertical veiw angle
        point3 look_from = point3(0, 0, 0); // Point camera looks from
        point3 look_at = point3(0, 0, -1);  // Point camera looks at
        point3 vup = point3(0, 1, 0);       // Camera relative up direction  
        
        float defocus_angle = 0.0f;         // Variation angle of rays through each pixel
        float focus_dist = 10.0f;           // Distance from camera lookfrom point to plane of perfect focus

        __host__ camera* move_to_device() {
            // First initialize on host
            initialize();

            // Then copy to device
            camera* d_cam;
            cudaMalloc(&d_cam, sizeof(camera));
            cudaMemcpy(d_cam, this, sizeof(camera), cudaMemcpyHostToDevice);
            return d_cam;
        }
        
        __device__ void make_ray(int i, int j, int pixel_idx, int idx, rayBuffer* rays, curandStatePhilox4_32_10_t* state){
            /* 
                Construct a camera ray originating from the defocus disk and directed at a randomly
                sampled point around the pixel location i, j.
            */

            auto offset = sample_square(state);
            auto pixel_sample = pixel00_loc + ((i + offset.x())*pixel_delta_u) + ((j + offset.y())*pixel_delta_v);

            auto ray_origin = (defocus_angle <= 0) ? center : defocus_disk_sample(state);
            auto ray_direction =  pixel_sample - ray_origin;
            int alive = 1;
        
    
            init_ray(ray_origin, ray_direction, alive, pixel_idx, rays, idx);
        }


    private:
    // Private camera parameters
        point3 center;              // Camera center
        point3 pixel00_loc;         // Location of pixel 00
        vec3 pixel_delta_u;         // Offset to pixel to the right
        vec3 pixel_delta_v;         // Offset to pixel below
        vec3 u, v, w;               // Camera frame basis vectors
        vec3 defocus_disk_u;        // Defocus disk horizontal radius
        vec3 defocus_disk_v;        // Defocus disk vertical radius

        __host__ void initialize(){
            /* Initialize the Viewport */
            
            // Calculate the image height and ensure that it is atleast 1

            center = look_from;

            // Compute viewport dimensions 
            float theta = degrees_to_radians(vfov);
            float h = tanf(theta / 2.0f);
            float viewport_height = 2.0f * h * focus_dist;
            float viewport_width = viewport_height * (float(width)/height);

            // Calculate the u, v, w, unit basis vectors for the camera coordinate frame.
            w = unit_vector(look_from-look_at);
            u = unit_vector(cross(vup, w));
            v = cross(w, u);

            // Calculate the vectors across the horizontal and down the vertical viewport edges
            vec3 viewport_u = viewport_width * u;
            vec3 viewport_v = viewport_height * -v;

            // Calculate the horizontal and vertical delta vectors from pixel to pixel
            pixel_delta_u = viewport_u/width;
            pixel_delta_v = viewport_v/height;

            // Calculate the location of the upper left pixel
            vec3 viewport_upper_left = center - (focus_dist * w) - viewport_u/2.0f - viewport_v/2.0f;
            pixel00_loc = viewport_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v);

            // Calculate the camera defocus disk basis vectors
            float defocus_radius = focus_dist * tanf(degrees_to_radians(defocus_angle / 2.0f));
            defocus_disk_u = u * defocus_radius;
            defocus_disk_v = v * defocus_radius;
        }



        __device__ vec3 sample_square(curandStatePhilox4_32_10_t* state) const {
            // return vector to a random point in the [-0.5, -0.5]- [0.5, 0.5] unit square
            return vec3(random_float(state) - 0.5f, random_float(state) - 0.5f, 0);
        }
        __device__ vec3 defocus_disk_sample(curandStatePhilox4_32_10_t* state) const {
            // Returns a random point in the camera defocus disk.
            auto p = random_on_unit_disk(state);
            return center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);

        }

};

#endif