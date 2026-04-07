#ifndef CAMERA_H
#define CAMERA_H

#include "hittable.h"
#include "material.h"
#include <omp.h>
#include <vector>

class camera {
    public:
        
        double aspect_ratio = 1.0;          // Ratio between width and height
        int image_width = 400;              // Image width in pixels
        int samples_per_pixel = 10;         // Count of random samples for each pixel
        int max_depth = 10;                 // Maximum number of recursions in ray_color

        double vfov = 90;                   // Vertical veiw angle
        point3 look_from = point3(0, 0, 0); // Point camera looks from
        point3 look_at = point3(0, 0, -1);  // Point camera looks at
        point3 vup = point3(0, 1, 0);       // Camera relative up direction  
        
        double defocus_angle = 0.0;         // Variation angle of rays through each pixel
        double focus_dist = 10;             // Distance from camera lookfrom point to plane of perfect focus

        __host__ camera* move_to_device() {
            // First initialize on host
            initialize();

            // Then copy to device
            camera* d_cam;
            cudaMalloc(&d_cam, sizeof(camera));
            cudaMemcpy(d_cam, this, sizeof(camera), cudaMemcpyHostToDevice);
            return d_cam;
        }
        
        __device__ ray get_ray(int i, int j, curandState* state){
            /* 
                Construct a camera ray originating from the defocus disk and directed at a randomly
                sampled point around the pixel location i, j.
            */

            auto offset = sample_square(state);
            auto pixel_sample = pixel00_loc + ((i + offset.x())*pixel_delta_u) + ((j + offset.y())*pixel_delta_v);

            auto ray_origin = (defocus_angle <= 0) ? center : defocus_disk_sample(state);
            auto ray_direction =  pixel_sample - ray_origin;

            return ray(ray_origin, ray_direction);
        }

        __device__ color ray_color(const ray& r, int depth, hittable** world, curandState* state) const {

            if (depth <= 0) return color(0,0,0); 
            hit_record rec;
           
            if ((*world) -> hit(r, interval(0.001, infinity), rec)){
                ray scattered;
                color attenuation;
                if (rec.mat -> scatter(r, rec, attenuation, scattered, state))
                    return attenuation * ray_color(scattered, depth - 1, world, state);

                vec3 direction = rec.normal + random_unit_vector(state);
                return 0.5 * ray_color(ray(rec.p, direction), depth - 1,  world, state);
            }
            

            vec3 unit_direction = unit_vector(r.direction());
            double a = 0.5 * (unit_direction.y() + 1.0);
            
            return (1.0 - a)*color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
        } 


    private:
    // Private camera parameters
        int image_height;           // Height of the image
        double pixel_sample_scale;  // Color scale factor for sum of pixel samples
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
            image_height = int(image_width/aspect_ratio);
            image_height = (image_height < 1) ? 1 : image_height;

            pixel_sample_scale = 1.0/samples_per_pixel;

            center = look_from;

            // Compute viewport dimensions 
            double theta = degrees_to_radians(vfov);
            double h = tan(theta / 2);
            double viewport_height = 2 * h * focus_dist;
            double viewport_width = viewport_height * (double(image_width)/image_height);

            // Calculate the u, v, w, unit basis vectors for the camera coordinate frame.
            w = unit_vector(look_from-look_at);
            u = unit_vector(cross(vup, w));
            v = cross(w, u);

            // Calculate the vectors across the horizontal and down the vertical viewport edges
            vec3 viewport_u = viewport_width * u;
            vec3 viewport_v = viewport_height * -v;

            // Calculate the horizontal and vertical delta vectors from pixel to pixel
            pixel_delta_u = viewport_u/image_width;
            pixel_delta_v = viewport_v/image_height;

            // Calculate the location of the upper left pixel
            vec3 viewport_upper_left = center - (focus_dist * w) - viewport_u/2 - viewport_v/2;
            pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

            // Calculate the camera defocus disk basis vectors
            double defocus_radius = focus_dist * tan(degrees_to_radians(defocus_angle / 2));
            defocus_disk_u = u * defocus_radius;
            defocus_disk_v = v * defocus_radius;
        }



        __device__ vec3 sample_square(curandState* state) const {
            // return vector to a random point in the [-0.5, -0.5]- [0.5, 0.5] unit square
            return vec3(random_double(state) - 0.5, random_double(state) - 0.5, 0);
        }
        __device__ vec3 defocus_disk_sample(curandState* state) const {
            // Returns a random point in the camera defocus disk.
            auto p = random_on_unit_disk(state);
            return center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);

        }

};

#endif