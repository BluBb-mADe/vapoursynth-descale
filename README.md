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

This fork generates the above-described matrices lazily (or reuses them from cache) when needed and caches them globally (instead of generating them during instantiation).<br>
This has three main advantages.
 - Matrices can be reused in case of multiple instances of descale inside the same vapoursynth context.
 - The Matrices for multiple filter instances with different parameters in the same context will be generated in the vapoursynth thread-pool whereas the original matrix creation happens on filter-tree generation itself and is therefore forced to run inside the single-threaded python (or whatever generates the filter-tree) context.
  - Matrices now can be used for either orientation which means that for example for an aspect ratio of 1:1 only 1 matrix would be needed.

[Getnative](https://github.com/Infiziert90/getnative) profits from the second and third point and the getnative functionality within the [Tsuzuru discord Bot](https://github.com/Infiziert90/Tsuzuru-Bot) additionally profits from the first point tremendously.

I've made some very crude benchmarks without spotting any performance differences in a normal use-case so there seems to be no obvious disadvantage to the original.

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
