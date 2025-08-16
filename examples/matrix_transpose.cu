// Matrix transpose kernel - good candidate for optimization
#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transpose(float *odata, const float *idata, int width, int height) {
    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    
    int index_in = xIndex + width * yIndex;
    int index_out = yIndex + height * xIndex;
    
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (xIndex < width && yIndex + i < height) {
            odata[index_out + i * height] = idata[index_in + i * width];
        }
    }
}