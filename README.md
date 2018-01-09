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


## Compilation

### Linux
```
g++ -std=c++17 -shared -fPIC -O2 descale.cpp -o libdescale.so

clang -std=c++1z -shared -fPIC -O2 -lstdc++ descale.cpp -o libdescale.so
```

### Cross-compilation for Windows
```
x86_64-w64-mingw32-g++ -std=c++17 -shared -fPIC -O2 descale.cpp -static-libgcc -static-libstdc++ -Wl,-Bstatic -lstdc++ -lpthread -Wl,-Bdynamic -o libdescale.dll

x86_64-w64-mingw32-clang++ -std=c++1z -shared -fPIC -O2 descale.cpp -static-libgcc -static-libstdc++ -Wl,-Bstatic -lstdc++ -lpthread -Wl,-Bdynamic -s -o libdescale.dll
```
I couldn't spot any problems with higher optimization settings but it seems to actually perform worse on some systems.<br>
There isn't a big difference between gcc and clang but clang seems to perform just a tiny bit better in general.
