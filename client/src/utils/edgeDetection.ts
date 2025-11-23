/**
 * Edge Detection Utilities
 * Based on Sobel operator for edge detection
 * Reference: https://en.wikipedia.org/wiki/Edge_detection
 */

/**
 * Sobel operator kernels for edge detection
 */
const SOBEL_X = [
  [-1, 0, 1],
  [-2, 0, 2],
  [-1, 0, 1],
];

const SOBEL_Y = [
  [-1, -2, -1],
  [0, 0, 0],
  [1, 2, 1],
];

/**
 * Convert image to grayscale
 */
function toGrayscale(imageData: ImageData): number[] {
  const data = imageData.data;
  const grayscale: number[] = [];
  
  for (let i = 0; i < data.length; i += 4) {
    // Weighted grayscale conversion
    const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    grayscale.push(gray);
  }
  
  return grayscale;
}

/**
 * Apply Sobel operator to detect edges
 */
function applySobel(grayscale: number[], width: number, height: number): number[] {
  const edges: number[] = new Array(width * height).fill(0);
  
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      let gx = 0;
      let gy = 0;
      
      // Apply Sobel kernels
      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          const idx = (y + ky) * width + (x + kx);
          const pixel = grayscale[idx];
          
          gx += pixel * SOBEL_X[ky + 1][kx + 1];
          gy += pixel * SOBEL_Y[ky + 1][kx + 1];
        }
      }
      
      // Calculate gradient magnitude
      const magnitude = Math.sqrt(gx * gx + gy * gy);
      edges[y * width + x] = magnitude;
    }
  }
  
  return edges;
}

/**
 * Detect edges in an image using Sobel operator
 * Returns a binary mask where edges are white (255) and non-edges are black (0)
 */
export function detectEdges(
  canvas: HTMLCanvasElement,
  threshold: number = 50
): ImageData | null {
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;

  try {
    // Get image data from canvas
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const width = imageData.width;
    const height = imageData.height;

    // Convert to grayscale
    const grayscale = toGrayscale(imageData);

    // Apply Sobel operator
    const edges = applySobel(grayscale, width, height);

    // Normalize and threshold
    // Use reduce instead of spread operator to avoid stack overflow for large arrays
    const maxEdge = edges.reduce((max, val) => Math.max(max, val), 0);
    const normalizedThreshold = maxEdge > 0 ? (threshold / 255) * maxEdge : threshold;

    // Create binary edge mask
    const edgeData = new ImageData(width, height);
    for (let i = 0; i < edges.length; i++) {
      const edgeValue = edges[i] > normalizedThreshold ? 255 : 0;
      const idx = i * 4;
      edgeData.data[idx] = edgeValue;     // R
      edgeData.data[idx + 1] = edgeValue; // G
      edgeData.data[idx + 2] = edgeValue; // B
      edgeData.data[idx + 3] = 255;       // A
    }

    return edgeData;
  } catch (error) {
    console.error('Error detecting edges:', error);
    return null;
  }
}

/**
 * Flood fill algorithm to fill region starting from a point
 * Used for mask drawing with edge detection assistance
 */
export function floodFill(
  imageData: ImageData,
  startX: number,
  startY: number,
  fillColor: { r: number; g: number; b: number; a: number },
  edgeMask?: ImageData,
  tolerance: number = 10
): ImageData {
  const width = imageData.width;
  const height = imageData.height;
  const data = imageData.data;
  const result = new ImageData(new Uint8ClampedArray(data), width, height);
  const edgeData = edgeMask ? edgeMask.data : null;

  // Get target color at start position
  const startIdx = (startY * width + startX) * 4;
  const targetR = data[startIdx];
  const targetG = data[startIdx + 1];
  const targetB = data[startIdx + 2];

  // Stack for flood fill
  const stack: Array<[number, number]> = [[startX, startY]];
  const visited = new Set<string>();

  while (stack.length > 0) {
    const [x, y] = stack.pop()!;
    const key = `${x},${y}`;

    if (visited.has(key)) continue;
    if (x < 0 || x >= width || y < 0 || y >= height) continue;

    const idx = (y * width + x) * 4;

    // Check if we hit an edge (if edge mask is provided)
    if (edgeData) {
      const edgeIdx = (y * width + x) * 4;
      if (edgeData[edgeIdx] > 128) {
        // Hit an edge, stop filling
        continue;
      }
    }

    // Check if color is similar to target
    const r = data[idx];
    const g = data[idx + 1];
    const b = data[idx + 2];

    const colorDiff = Math.sqrt(
      Math.pow(r - targetR, 2) +
      Math.pow(g - targetG, 2) +
      Math.pow(b - targetB, 2)
    );

    if (colorDiff > tolerance) continue;

    // Fill the pixel
    result.data[idx] = fillColor.r;
    result.data[idx + 1] = fillColor.g;
    result.data[idx + 2] = fillColor.b;
    result.data[idx + 3] = fillColor.a;

    visited.add(key);

    // Add neighbors
    stack.push([x + 1, y]);
    stack.push([x - 1, y]);
    stack.push([x, y + 1]);
    stack.push([x, y - 1]);
  }

  return result;
}

/**
 * Get edge points near a given position
 * Useful for snapping mask drawing to edges
 */
export function getNearbyEdges(
  edgeMask: ImageData,
  x: number,
  y: number,
  radius: number = 10
): Array<{ x: number; y: number }> {
  const width = edgeMask.width;
  const height = edgeMask.height;
  const data = edgeMask.data;
  const edges: Array<{ x: number; y: number }> = [];

  const startX = Math.max(0, Math.floor(x - radius));
  const endX = Math.min(width - 1, Math.floor(x + radius));
  const startY = Math.max(0, Math.floor(y - radius));
  const endY = Math.min(height - 1, Math.floor(y + radius));

  for (let py = startY; py <= endY; py++) {
    for (let px = startX; px <= endX; px++) {
      const idx = (py * width + px) * 4;
      if (data[idx] > 128) {
        // Found an edge pixel
        edges.push({ x: px, y: py });
      }
    }
  }

  return edges;
}

