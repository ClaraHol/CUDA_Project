#ifndef BVH_H
#define BVH_H

#include "aabb.h"
#include "hittable.h"
#include "hittable_list.h"

/*
    This is the bounding volume hiarchy code. It is built on the 
    "Ray tracing the next week" book, so it is not the most efficient way to do BVH
    on the GPU, but it is simple.

    It consists of two classes:
    - bvh_node: The TRtnW BVH tree implementation.
    - bvh_array: A flattened array of the bvh_node which stores pointers to the objects.
    
    And two functions:
    - flatten: Turns the bvh_node recursion tree into the bvh_array.
    - build_bvh: Builds the BVH tree on the device and flattens it.

    Since the code is translated from CPU code not meant to run in parallel it only 
    runs on one thread. 
*/

__host__ __device__ void insertion_sort(hittable** objects, size_t start,
                                         size_t end,
                                         bool (*cmp)(const hittable*,
                                                     const hittable*)) {
    /* 
        Sorting algorithm for the bvh_node class which 
        can be used on the device. Not the best, but fine for small arrays. 
    */                                                   
    for (size_t i = start + 1; i < end; i++) {
        hittable* key = objects[i];
        int j = i - 1;
        while (j >= (int)start && cmp(key, objects[j])) {
            objects[j + 1] = objects[j];
            j--;
        }
        objects[j + 1] = key;
    }
}

class bvh_array {
    // Flat array version of the BVH tree to use on GPU
    public:
        aabb bbox;
        int left;
        int right;
        int object_index;

        __device__ bool hit(bvh_array* nodes, hittable** objects, const ray& r, interval ray_t, hit_record& rec) const {

            int stack[32];
            int stack_top = 0;
            stack[stack_top++] = 0;

            bool hit_anything = false;
            while (stack_top > 0){
                // Access the root node
                bvh_array& node = nodes[stack[--stack_top]];

                if (!node.bbox.hit(r, ray_t)) continue;

                if (node.object_index >= 0){
                    if (objects[node.object_index]->hit(r, ray_t, rec)){
                        hit_anything = true;
                        ray_t.max = rec.t;
                    } 
                } else {
                    stack[stack_top++] = node.left;
                    stack[stack_top++] = node.right;
                }
            }

            return hit_anything;
        }
};  


class bvh_node : public hittable {
    public:
        bool is_leaf;

        __device__ bvh_node(hittable_list& list, size_t size) : bvh_node(list.objects, 0, size) {}

        __device__ bvh_node(hittable** objects, size_t start, size_t end){
            // Constructing the BVH

                bbox = empty_aabb();
                for (size_t object_index = start; object_index < end; object_index++){
                    bbox = aabb(bbox, objects[object_index]->bounding_box());
                }

                int axis = bbox.longest_axis();
                auto comparator = (axis == 0) ? box_x_compare : (axis == 1) ? box_y_compare : box_z_compare;
                size_t span = end - start;

                if (span == 1){
                    // Leaf node
                    left = right = objects[start];
                    is_leaf = true;
                } else if (span == 2) {
                    // Internal node with 2 childen
                    left = objects[start];
                    right = objects[start + 1];
                    is_leaf = false;
                } else {
                    // Many children, make new nodes
                    insertion_sort(objects + start, 0, span, comparator);

                    auto mid = start + span/2;
                    left = new bvh_node(objects, start, mid);
                    right = new bvh_node(objects, mid, end);
                    is_leaf = false;
                }
            
        }
        __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
            return false; // bvh_node is only used during construction, never traversed directly
        }

        __device__ aabb bounding_box() const override { return bbox; }
        
        __device__ friend int flatten(hittable* node, bvh_array* flat, hittable** objects, int num_objects, int& index);
        
    private:
        hittable* left;
        hittable* right;
        aabb bbox;

        __device__ static bool box_compare(const hittable* a, const hittable* b, int axis_idx){
            auto a_axis_interval = a->bounding_box().axis_interval(axis_idx);
            auto b_axis_interval = b->bounding_box().axis_interval(axis_idx);
            return a_axis_interval.min < b_axis_interval.min;
        }

        __device__ static bool box_x_compare(const hittable* a, const hittable* b){
            return box_compare(a, b, 0);
        }

        __device__ static bool box_y_compare(const hittable* a, const hittable* b){
            return box_compare(a, b, 1);
        }
        
        __device__ static bool box_z_compare(const hittable* a, const hittable* b){
            return box_compare(a, b, 2);
        }
};

__device__ int flatten(hittable* node, bvh_array* flat,
                     hittable** objects, int num_objects, int& index) {

    /* 
        Flatten the the recursive tree to use on the GPU. 
        The array is stored in the bvh_array class.
        
    */

    int my_index = index++;
    flat[my_index].bbox = node->bounding_box();


    bool is_leaf = false;
    for (int i = 0; i < num_objects; i++) {
        if (objects[i] == node) {
            flat[my_index].object_index = i;
            flat[my_index].left  = -1;
            flat[my_index].right = -1;
            is_leaf = true;
            break;
        }
    }

    if (!is_leaf) {
        // If bvh is not a leaf, make a new node
        bvh_node* bvh = static_cast<bvh_node*>(node);
        flat[my_index].object_index = -1;
        flat[my_index].left  = flatten(bvh->left,  flat, objects, num_objects, index);
        flat[my_index].right = flatten(bvh->right, flat, objects, num_objects, index);
    }
    return my_index;
}



__global__ void build_bvh(hittable** objects, int num_objects, bvh_array* flat, int* flat_size){
    if (threadIdx.x == 0 && blockIdx.x == 0){

        // Build BVH on device using one thread
        bvh_node* root = new bvh_node(objects, 0, num_objects);

        // Flatten into bvh_array
        int idx = 0;
        flatten(root, flat, objects, num_objects, idx);

        *flat_size = idx;
        // Clean up
        //delete root;
    }
}

#endif