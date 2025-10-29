#include "shaders.h"

namespace Shaders
{

    const char *vertexShaderSource = R"(
    #version 330 core
    // Vertex attributes
    layout (location = 0) in vec2 aPos;
    // Texture coordinates
    layout (location = 1) in vec2 aTexCoord;
    // Output to fragment shader
    out vec2 TexCoord;

    // Transformation uniforms
    uniform vec2 uTranslate;
    uniform float uScale;
    uniform float uRotation; // Rotation in radians

    void main() {
        // Create rotation matrix
        float cosTheta = cos(uRotation);
        float sinTheta = sin(uRotation);
        // 2x2 rotation matrix
        mat2 rotationMatrix = mat2(
            cosTheta, -sinTheta,
            sinTheta, cosTheta
        );
        
        // Apply transformations: scale -> rotate -> translate
        vec2 pos = aPos * uScale;
        pos = rotationMatrix * pos;
        pos = pos + uTranslate;
        
        // Set final vertex position
        gl_Position = vec4(pos.x, pos.y, 0.0, 1.0);
        TexCoord = aTexCoord;
    }
)";

    const char *fragmentShaderSource = R"(
    #version 330 core
    // Output color
    out vec4 FragColor;
    // Input texture coordinates
    in vec2 TexCoord;
    // Texture sampler
    uniform sampler2D ourTexture;
    void main() {
        FragColor = texture(ourTexture, TexCoord);
    }
)";

    const char *grayscaleShaderSource = R"(
    #version 330 core
    // Output color
    out vec4 FragColor;
    // Input texture coordinates
    in vec2 TexCoord;
    // Texture sampler
    uniform sampler2D ourTexture;
    void main() {
        // Fetch the color from texture
        vec4 color = texture(ourTexture, TexCoord);
        // Convert to grayscale using luminosity method
        float gray = dot(color.rgb, vec3(0.299, 0.587, 0.114));
        // Output the grayscale color
        FragColor = vec4(vec3(gray), 1.0);
    }
)";

    const char *blurShaderSource = R"(
    #version 330 core
    // Output color
    out vec4 FragColor;
    // Input texture coordinates
    in vec2 TexCoord;
    // Texture sampler
    uniform sampler2D ourTexture;
    // Texel size for sampling neighboring pixels
    uniform vec2 texelSize;
    
    void main() {
        vec4 result = vec4(0.0);
        
        // 5x5 Gaussian kernel weights
        float kernel[25] = float[](
            1.0/273.0, 4.0/273.0, 7.0/273.0, 4.0/273.0, 1.0/273.0,
            4.0/273.0, 16.0/273.0, 26.0/273.0, 16.0/273.0, 4.0/273.0,
            7.0/273.0, 26.0/273.0, 41.0/273.0, 26.0/273.0, 7.0/273.0,
            4.0/273.0, 16.0/273.0, 26.0/273.0, 16.0/273.0, 4.0/273.0,
            1.0/273.0, 4.0/273.0, 7.0/273.0, 4.0/273.0, 1.0/273.0
        );
        
        // Applying the kernel
        int index = 0;
        for(int y = -2; y <= 2; y++) {
            for(int x = -2; x <= 2; x++) {
                // Calculate offset
                vec2 offset = vec2(x, y) * texelSize;
                // Accumulate weighted color
                result += texture(ourTexture, TexCoord + offset) * kernel[index++];
            }
        }
        FragColor = result;
    }
)";

    const char *edgeShaderSource = R"(
    #version 330 core
    // Output color
    out vec4 FragColor;
    // Input texture coordinates
    in vec2 TexCoord;
    // Texture sampler
    uniform sampler2D ourTexture;
    // Texel size for sampling neighboring pixels
    uniform vec2 texelSize;
    
    void main() {
        // Sobel kernels for edge detection
        float Gx[9] = float[](-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0);
        float Gy[9] = float[](-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0);
        
        // Accumulators for gradients
        float edgeX = 0.0;
        float edgeY = 0.0;
        int index = 0;
        
        // Applying Sobel operator
        for(int y = -1; y <= 1; y++) {
            for(int x = -1; x <= 1; x++) {
                // Calculate offset
                vec2 offset = vec2(x, y) * texelSize;
                // Fetch color and convert to grayscale
                vec4 color = texture(ourTexture, TexCoord + offset);
                // Luminosity method
                float gray = dot(color.rgb, vec3(0.299, 0.587, 0.114));
                // Accumulate gradients
                edgeX += gray * Gx[index];
                edgeY += gray * Gy[index];
                index++;
            }
        }
        
        // Calculate gradient magnitude
        float edge = sqrt(edgeX * edgeX + edgeY * edgeY);
        // Threshold to make edges more visible (similar to Canny)
        edge = edge > 0.3 ? 1.0 : 0.0;
        FragColor = vec4(vec3(edge), 1.0);
    }
)";

    const char *pixelationShaderSource = R"(
    #version 330 core
    // Output color
    out vec4 FragColor;
    // Input texture coordinates
    in vec2 TexCoord;
    // Texture sampler
    uniform sampler2D ourTexture;
    // Pixelation size
    uniform float pixelSize;
    // Resolution of the texture
    uniform vec2 resolution;
    
    void main() {
        // Calculate pixel block size
        vec2 blockSize = vec2(pixelSize) / resolution;
        
        // Snap to pixel grid
        vec2 coord = floor(TexCoord / blockSize) * blockSize;
        
        FragColor = texture(ourTexture, coord);
    }
)";

    const char *comicShaderSource = R"(
    #version 330 core
    // Output color
    out vec4 FragColor;
    // Input texture coordinates
    in vec2 TexCoord;
    // Texture sampler
    uniform sampler2D ourTexture;
    // Texel size for sampling neighboring pixels
    uniform vec2 texelSize;
    
    void main() {
        // Edge detection using Sobel
        float Gx[9] = float[](-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0);
        float Gy[9] = float[](-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0);
        
        // Accumulators for gradients
        float edgeX = 0.0;
        float edgeY = 0.0;
        int index = 0;
        
        // Applying Sobel operator
        for(int y = -1; y <= 1; y++) {
            for(int x = -1; x <= 1; x++) {
                vec2 offset = vec2(x, y) * texelSize;
                vec4 color = texture(ourTexture, TexCoord + offset);
                float gray = dot(color.rgb, vec3(0.299, 0.587, 0.114));
                edgeX += gray * Gx[index];
                edgeY += gray * Gy[index];
                index++;
            }
        }
        
        // Calculate gradient magnitude
        float edge = sqrt(edgeX * edgeX + edgeY * edgeY);
        // Create edge mask
        float edgeMask = edge > 0.2 ? 0.0 : 1.0;
        
        // Color quantization (reduce colors like comic art)
        vec4 color = texture(ourTexture, TexCoord);
        // Reduce to limited color palette
        float levels = 4.0; // Number of color levels
        // Quantize color
        vec3 quantized = floor(color.rgb * levels) / levels + 0.5 / levels;
        
        // Combine edges with quantized color
        FragColor = vec4(quantized * edgeMask, 1.0);
    }
)";
}