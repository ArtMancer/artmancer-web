/**
 * Canvas Layer Constants
 * Định nghĩa các constants cho z-index và layer names
 */

export const Z_INDEX = {
    IMAGE: 10,
    MASK: 20,
    UI: 30,
    DIVIDER: 40,
    LOADING: 50,
} as const;

export enum LAYER_NAMES {
    ROOT = 'root',
    VIEWPORT = 'viewport',
    IMAGE_CONTAINER = 'image-container',
    CONTENT = 'content',
    TRANSFORM = 'transform',
    IMAGE = 'image',
    MASK = 'mask',
    UI_OVERLAY = 'ui-overlay',
}

