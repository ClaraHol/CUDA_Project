#ifndef SCENE_H
#define SCENE_H

#include "rt_weekend.h"
#include "hittable.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"

__global__ void create_world(hittable** d_list, hittable** d_world, material** d_material){
    /* 
        Create the Scene. Note that this has to be changed to change the scene
    */

    // Define world on host using one thread only
    if (threadIdx.x == 0 && blockIdx.x == 0){
        // Define materials
        d_material[0] = new lambertian(color(0.5, 0.5, 0.5));   // Center ball
        d_material[1] = new metal(color(0.1, 0.2, 0.5), 0.1);   // Ground material
        d_material[2] = new dielectic(1.5);                     // Hollow Glass ball 
        d_material[3] = new dielectic(1.00/1.5);                // Air inside Glass ball
        d_material[4] = new metal(color(0.8, 0.6, 0.2), 0.8);   // Right metal ball (matte)


        // Add spheres to world (sphere(center, radius, material))
        d_list[0] = new sphere(point3(0, -100.5, -1), 100, d_material[1]);
        d_list[1] = new sphere(point3(0, 0, -1.2), 0.5, d_material[0]);
        d_list[2] = new sphere(point3(-1.0, 0.0, -1.0), 0.5, d_material[2]);
        d_list[3] = new sphere(point3(-1.0, 0.0, -1.0), 0.4, d_material[3]);
        d_list[4] = new sphere(point3(1.0, 0.0, -1.0), 0.5, d_material[4]);

        // Create world
        *d_world = new hittable_list(d_list, 3)
    }
};

__global__ void destroy_world(hittable** d_list, hittable** d_world, material** d_material, int num_objects){
    if (threadIdx.x == 0 && blockIdx.x == 0){
        for (int i; i < num_objects, i++){
            delete d_list[i];
            delete d_material[i];
        }
        
        delete *d_world;
    }
};


#endif