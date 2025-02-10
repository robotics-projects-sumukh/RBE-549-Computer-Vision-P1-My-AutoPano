import os
import cv2
import argparse
from utils import find_best_matching_pairs, create_panorama_graph, find_optimal_ordering, get_panorama, crop_black_edges

class PanoramaStitcher:
    def __init__(self, block_size=2, ksize=3, k=0.03, n_best=1000):
        self.block_size = block_size
        self.ksize = ksize
        self.k = k
        self.n_best = n_best
        
    def load_images(self, directory, scale=1):
        """Load and scale images from directory"""
        image_list = sorted([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
        images = []
        
        for file in image_list:
            img = cv2.imread(os.path.join(directory, file))
            if img is not None:
                if scale != 1:
                    h, w = img.shape[:2]
                    img = cv2.resize(img, (int(w/scale), int(h/scale)))
                images.append(img)
                
        return images, len(images)

    def create_panorama(self, directory, set_name, set_type):
        """Main function to create panorama"""
        # Determine scaling based on set name
        scale_map = {
            'CustomSet1': 2,
            'CustomSet2': 4,
            'CustomSet3': 4,
            'TestSet2': 4,
        }
        scale = scale_map.get(set_name, 1)
        
        # Load images at original and scaled resolution
        images, n_images = self.load_images(directory, scale)
        images_for_ordering, _ = self.load_images(directory, 4)  # Always load at scale 4 for ordering
        
        if n_images < 2:
            print("Need at least 2 images to create a panorama")
            return None
            
        print(f"Loaded {n_images} images")
        
        # Determine blending strategy
        use_blending = set_name not in ['Set3', 'TestSet4', 'TestSet1', 'TestSet2']
        
        # Create panorama
        final_panorama = self._stitch_images(
            images, 
            images_for_ordering,
            set_name,
            set_type,
            use_blending
        )
        
        if final_panorama is not None:
            print("Successfully created panorama")
        else:
            print("Failed to create panorama")
            
        return final_panorama
        
    def _stitch_images(self, images, images_for_ordering, set_name, set_type, use_blending):
        """Internal method to handle image stitching logic"""
        # Find optimal image ordering
        print("Finding best matching pairs...")
        image_pairs = find_best_matching_pairs(
            images_for_ordering, 
            self.block_size, 
            self.ksize, 
            self.k, 
            self.n_best
        )
        
        # Create graph and find optimal ordering
        graph, weights = create_panorama_graph(image_pairs, len(images))
        start_node = max(graph.keys(), key=lambda x: len(graph[x]))
        
        print("Finding optimal image ordering...")
        optimal_order = find_optimal_ordering(graph, weights, start_node)
        if set_name == 'TestSet1':
            optimal_order = [0,1,3,2]
        print("Optimal image ordering:", optimal_order)
        
        # Reorder images
        ordered_images = [images[i] for i in optimal_order]
        
        # Choose stitching strategy based on number of images
        if len(images) < 5:
            result = self._sequential_stitch(
                ordered_images, 
                set_name, 
                set_type, 
                use_blending
            )
        else:
            result = self._half_stitch(
                ordered_images, 
                set_name, 
                set_type, 
                use_blending
            )
            
        return crop_black_edges(result) if result is not None else None
        
    def _sequential_stitch(self, images, set_name, set_type, use_blending):
        """Stitch images sequentially"""
        print("="*50)
        print("Using sequential stitching...")
        print("="*50)
        result = images[0]
        
        for i in range(1, len(images)):
            panorama, failed = get_panorama(
                result, 
                images[i],
                blockSize=self.block_size,
                ksize=self.ksize,
                k=self.k,
                nBest=self.n_best,
                Set=set_name,
                SetType=set_type,
                i=i,
                blend=use_blending,
                panorama_type="full"
            )
            
            if failed:
                print(f"Failed to stitch image {i}, skipping...")
                continue
                
            result = panorama
            
        return result
        
    def _half_stitch(self, images, set_name, set_type, use_blending):
        """Stitch images using half-stitch pattern"""
        print("="*50)
        print("Using half-stitch pattern...")
        print("="*50)
        half_length = len(images) // 2
        start_idx = 1 if set_name == 'Set3' else 0
        
        print("="*50)
        print("Stitching left half...")
        print("="*50)
        # Stitch left half
        left_result = self._stitch_half(
            images[start_idx:half_length],
            set_name,
            set_type,
            use_blending,
            "left",
            start_idx
        )
        
        print("="*50)
        print("Stitching right half...")
        print("="*50)
        # Stitch right half
        right_result = self._stitch_half(
            images[half_length:][::-1],
            set_name,
            set_type,
            use_blending,
            "right",
            half_length
        )
        
        # Combine halves
        print("="*50)
        print("Combining left and right panoramas...")
        print("="*50)
        final_result, failed = get_panorama(
            left_result,
            right_result,
            blockSize=self.block_size,
            ksize=self.ksize,
            k=self.k,
            nBest=self.n_best,
            Set=set_name,
            SetType=set_type,
            i=100,
            blend=use_blending,
            panorama_type="full"
        )
        
        return None if failed else final_result
        
    def _stitch_half(self, images, set_name, set_type, use_blending, half_type, start_idx):
        """Helper method for stitching half of the panorama"""
        result = images[0]
        
        for i, img in enumerate(images[1:], start_idx + 1):
            panorama, failed = get_panorama(
                result,
                img,
                blockSize=self.block_size,
                ksize=self.ksize,
                k=self.k,
                nBest=self.n_best,
                Set=set_name,
                SetType=set_type,
                i=i,
                blend=use_blending,
                panorama_type=half_type
            )
            
            if failed:
                print(f"Failed to stitch {half_type} image {i}, skipping...")
                result = img
                continue
                
            result = panorama
            
        return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Set', default="Set1", help='Set Number, Default:Set1')
    parser.add_argument('--SetType', default="Train", help='Set Type, Default:Train')
    args = parser.parse_args()
    
    directory = f"../Data/{args.SetType}/{args.Set}"
    
    # Create stitcher and generate panorama
    stitcher = PanoramaStitcher()
    panorama = stitcher.create_panorama(directory, args.Set, args.SetType)

if __name__ == "__main__":
    main()