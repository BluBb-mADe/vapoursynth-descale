# Descale

VapourSynth plugin to undo upscaling.

## Usage

The plugin itself only supports GrayS, RGBS, and YUV444PS input.
The included python wrapper supports YUV (every subsampling), Gray, and RGB of every bitdepth.

##### Descalers:
```
descale.Debilinear(clip src, int width, int height, float src_left=0.0, float src_top=0.0)

descale.Debicubic(clip src, int width, int height, float b=1/3, float c=1/3, float src_left=0.0, float src_top=0.0)

descale.Delanczos(clip src, int width, int height, int taps=3, float src_left=0.0, float src_top=0.0)

descale.Despline16(clip src, int width, int height, float src_left=0.0, float src_top=0.0)

descale.Despline36(clip src, int width, int height, float src_left=0.0, float src_top=0.0)

descale.Despline36(clip src, int width, int height, int kernel="bilinear" float src_left=0.0, float src_top=0.0)
```
##### CacheSize:
`descale.CacheSize(int size=-1)` changes the internal size of the matrix lru cache. -1 (default) means no limit.
 

## How does this work?

Resampling can be described as `A x = b`.

A is an n x m matrix with m being the input dimension and n the output dimension. x is the original vector with m elements, b is the vector after resampling with n elements. We want to solve this equation for x.

To do this, we extend the equation with the transpose of A: `A' A x = A' b`.

`A' A` is now a banded symmetrical m x m matrix and `A' b` is a vector with m elements.

This enables us to use LDLT decomposition on `A' A` to get `LD L' = A' A`. LD and L are both triangular matrices.

Then we solve `LD y = A' b` with forward substitution, and finally `L' x = y` with back substitution.

We now have the original vector `x`.

## Differences of this Fork to the Original

This fork generates the above-described matrices lazily (or reuses them from cache) when needed (instead of generating them during instantiation) and caches them globally. The use-cases of this improvement are quite specific. The main benefit is that matrices can be reused in case of multiple instances of descale inside the same vapoursynth context. The second benefit is that if we use different resolutions in different instances in the same context they will be generated in the vapoursynth thread-pool whereas the original matrix creation happened on filter tree generation and is forced to run inside the single-threaded python (or whatever generates the filter-tree) context. Another (mostly irrelevant) benefit is that matrices now can be used for either orientation which means that for example for an aspect ratio of 1:1 only 1 matrix would be needed.

I've made some very crude benchmarks without spotting any performance differences in a normal use-case so there seems to be no disadvantage to the original.

An example of how this can drastically improve performance is getnative and especially the getnative functionality within the Tsuzuru discord bot.

## Compilation

### Linux
```
g++ -std=c++17 -shared -fPIC -O2 descale.cpp -o libdescale.so
```

### Cross-compilation for Windows
```
x86_64-w64-mingw32-g++ -std=c++17 -shared -O2 descale.cpp -static-libgcc -Wl,-Bstatic -lstdc++ -lpthread -s -o libdescale.dll
```
I couldn't spot any problems with higher optimization settings but it seems to actually perform worse on some systems.
