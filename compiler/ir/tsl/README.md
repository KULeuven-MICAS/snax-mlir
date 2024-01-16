# TSL Memory Layouts

## Rationale

A TSL (tiled-strided-layout) memory layout is an MLIR attribute, designed to be used as the `layout` parameter for a `memref` type. A TSL layout tiles the data and defines a stride for every tile, allowing for flexible memory layouts especially suited for hardware accelerators. This layout adds tiling to the existing `StridedLayoutAttr`. While the `AffineMapLayoutAttr` allows for a tiled layout, the representation is not always clear, and more importantly does not allow for non-contiguity, which may be required to maximally exploit the full bandwidth of the memory.

## Notation

We employ the following notation for TSL attributes: (for a 2D matrix and one level of tiling), where the bounds and strides are ordered from outermost -> innermost

`[bound, bound] -> (stride, stride), [bound, bound] -> (stride, stride)`

Consider the following memory layout:
The image represents an `8x8` matrix, where every digit represents the memory address where the element will be stored.

`<img src="https://github.com/KULeuven-MICAS/snax-mlir/assets/47864363/6d03debe-888e-4e5f-82c2-040434bc1f99 " width="400">`

In both dimensions, the data is tiled in 2 tiles of size 4, this information is represented with the tiling bounds:

`[2, 4] -> (stride, stride), [2, 4] -> (stride, stride)`

For the first dimension there is a stride within the tile of 4 and across tiles of 32:

`[2, 4] -> (32, 4), [4, 2] -> (stride, stride)`

For the second dimension there is a stride within the tile of 1 and across tiles of 16:

`[2, 4] -> (32, 4), [2, 4] -> (16, 1)`

Additionally, the full TSL layout attribute can also include a base memory offset:

`#tsl.tsl<[2, 4] -> (32, 4), [2, 4] -> (16, 1), offset: 5>`

When no offset is defined, it is assumed to be 0

## Dynamic Sizes

The layout provided allows for some flexibility in defining dynamic shapes within a matrix:

`#tsl.tsl<[?, 4] -> (32, 4), [?, 4] -> (?, 1)>`

The key point is that only the outermost tile is allowed to have dynamic sizes; the sizes and strides of the inner tiles must remain fixed. In the example, the fixed tile sizes are set to 4x4, with strides of 4 and 1. Additionally, there's one extra stride of 32, causing the tiles to be spaced at intervals of 32. The determination of the other strides, once the full matrix dimensions are known, is not yet determined. However, a likely approach is to densely determine the strides from left to right.

For example, if dealing with a 64x64 matrix, the layout would be adjusted accordingly:

`#tsl.tsl<[16, 4] -> (32, 4), [16, 4] -> (?, 1)>`

Here, the missing stride is calculated as 32x16=512. This adjustment ensures that the dynamic shapes remain consistent with the fixed tile sizes and strides while accommodating the overall matrix dimensions.
