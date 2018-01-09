/*
 * Copyright © 2017 Frechdachs <frechdachs@rekt.cc>
 * This program is free software. It comes without any warranty, to
 * the extent permitted by applicable law. You can redistribute it
 * and/or modify it under the terms of the Do What The Fuck You Want
 * To Public License, Version 2, as published by Sam Hocevar.
 * See the COPYING file for more details.
 */


#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <shared_mutex>
#include "vapoursynth/VapourSynth.h"
#include "vapoursynth/VSHelper.h"

using namespace std;


enum DescaleMode : int
{
    bilinear = 0,
    bicubic  = 1,
    lanczos  = 2,
    spline16 = 3,
    spline36 = 4
};

struct DescaleKey {
    char key[24];
    int refcount = 1;
    int offset = 0;
    int src;
    int dst;
    float shift;
    unsigned int hash = 2166136261u;

    // this weird and inconsistent struct initialization came into being because of circular
    // type dependencies between DescaleKey and DescaleData and desperate optimization.
    // see init_keys() in DescaleData for the main part of the initialization.
    DescaleKey(int src, int dst, float shift){
        this->src = src;
        this->dst = dst;
        this->shift = shift;
        add_var((short)src);
        add_var((short)dst);
        add_var(shift);
    }

    void init_hash(){
        for (char i : key) {
            hash ^= i;
            hash *= 16777619u;
        }
    }

    template<typename T>
    void add_var(T t) {
        memcpy(key+offset, &t, sizeof(t));
        offset += sizeof(t);
    }
};

struct DescaleKeyHash {
    size_t operator()(const DescaleKey* k) const {
        return k->hash;
    }
};

struct DescaleKeyEqual {
    bool operator()(const DescaleKey* lhs, const DescaleKey* rhs) const {
        return memcmp(lhs->key, rhs->key, 24) == 0;
    }
};

struct DescaleData {
    double b, c;
    float shift_h, shift_v;
    int taps;
    int support;
    int bandwidth;
    DescaleMode mode;
    VSVideoInfo vi;
    VSVideoInfo vi_dst;
    VSNodeRef *node;
    DescaleKey* hk;
    DescaleKey* vk;

    void init_keys() {
        hk = new DescaleKey(vi.width, vi_dst.width, shift_h);
        vk = new DescaleKey(vi.height, vi_dst.height, shift_v);
        hk->add_var((float)this->b);
        hk->add_var((float)this->c);
        hk->add_var((short)this->taps);
        hk->add_var((short)this->support);
        hk->add_var((short)this->bandwidth);
        hk->add_var((short)this->mode);
        // premature optimization right here, look at it, isn't it beautifully ugly?
        memcpy(&vk->key[vk->offset], &hk->key[vk->offset], hk->offset - vk->offset);
        hk->init_hash();
        vk->init_hash();
    }
};

struct Matrix {
    vector<float> upper;
    vector<float> diagonal;
    vector<float> lower;
    vector<float> weights;
    vector<int> weights_left_idx;
    vector<int> weights_right_idx;
    shared_mutex lock;
    bool ready=false;
};

struct Node {
    DescaleKey* key;
    Matrix* value;
    Node *prev, *next;
};

class DoublyLinkedList {
public:
    Node *front, *back;
    mutex list_lock;
    Node* add_page_to_head(DescaleKey* key, Matrix* value) {
        auto *page = new Node{key, value};
        lock_guard<mutex>lock(list_lock);
        if(!front && !back) {
            front = back = page;
        }
        else {
            page->next = front;
            front->prev = page;
            front = page;
        }
        return page;
    }

    void move_page_to_head(Node* page) {
        lock_guard<mutex>lock(list_lock);
        if(page == front) {
            return;
        }
        if(page == back) {
            back = back->prev;
            back->next = nullptr;
        }
        else {
            page->prev->next = page->next;
            page->next->prev = page->prev;
        }

        page->next = front;
        page->prev = nullptr;
        front->prev = page;
        front = page;
    }

    DescaleKey * remove_back_page() {
        DescaleKey * k;
        lock_guard<mutex>lock(list_lock);
        if(front == back) {
            k = back->key;
            delete back;
            front = back = nullptr;
        }
        else {
            k = back->key;
            delete back;
            back = back->prev;
            back->next = nullptr;
        }
        return k;
    }
};

int max_cache_size = -1;
DoublyLinkedList cacheList;
unordered_map<DescaleKey*, Node*, DescaleKeyHash, DescaleKeyEqual> cacheMap = unordered_map<DescaleKey*, Node*, DescaleKeyHash, DescaleKeyEqual>();
shared_mutex cache_lock;


static vector<double> transpose_matrix(int rows, const vector<double> &matrix)
{
    int columns = matrix.size() / rows;
    vector<double> transposed_matrix (matrix.size(), 0);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            transposed_matrix[i + rows * j] = matrix[i * columns + j];
        }
    }

    return transposed_matrix;
}


static vector<double> multiply_sparse_matrices(int rows, const vector<int> &lidx, const vector<int> &ridx, const vector<double> &lm, const vector<double> &rm)
{
    int columns = lm.size() / rows;
    vector<double> multiplied (rows * rows, 0);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < rows; ++j) {
            double sum = 0;

            for (int k = lidx[i]; k < ridx[i]; ++k) {
                sum += lm[i * columns + k] * rm[k * rows + j];
            }

            multiplied[i * rows + j] = sum;
        }
    }

    return multiplied;
}


static void multiply_banded_matrix_with_diagonal(int rows, int bandwidth, vector<double> &matrix)
{
    int c = (bandwidth + 1) / 2;

    for (int i = 1; i < rows; ++i) {
        int start = max(i - (c - 1), 0);
        for (int j = start; j < i; ++j) {
            matrix[i * rows + j] *= matrix[j * rows + j];
        }
    }
}


// LDLT decomposition (variant of Cholesky decomposition)
// Input is only the upper part of a banded symmetrical matrix in compressed form.
// The input matrix is modified in-place and contains L' and D in compressed form
// after decomposition. The main diagonal of ones of L' is not saved.
static void banded_ldlt_decomposition(int rows, int bandwidth, vector<double> &matrix)
{
    int c = (bandwidth + 1) / 2;
    // Division by 0 can happen if shift is used
    double eps = numeric_limits<double>::epsilon();

    for (int k = 0; k < rows; ++k) {
        int last = min(k + c - 1, rows - 1) - k;

        for (int j = 1; j <= last; ++j) {
            int i = k + j;
            double d = matrix[k * c + j] / (matrix[k * c] + eps);

            for (int l = 0; l <= last - j; ++l) {
                matrix[i * c + l] -= d * matrix[k * c + j + l];
            }
        }

        double e = 1.0 / (matrix[k * c] + eps);
        for (int j = 1; j < c; ++j) {
                matrix[k * c + j] *= e;
        }
    }
}


static vector<double> compress_matrix(int rows, const vector<int> &lidx, const vector<int> &ridx, const vector<double> &matrix)
{
    int columns = matrix.size() / rows;
    int max = 0;

    for (int i = 0; i < lidx.size(); ++i) {
        if (ridx[i] - lidx[i] > max)
            max = ridx[i] - lidx[i];
    }

    vector<double> compressed (rows * max, 0);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < ridx[i] - lidx[i]; ++j) {
            compressed[i * max + j] = matrix[i * columns + lidx[i] + j];
        }
    }

    return compressed;
}


static vector<double> compress_symmetric_banded_matrix(int rows, int bandwidth, const vector<double> &matrix)
{
    int c = (bandwidth + 1) / 2;
    vector<double> compressed (rows * c, 0);

    for (int i = 0; i < rows; ++i) {
        if (i < rows - c - 1) {

            for (int j = i; j < c + i; ++j) {
                compressed[i * c + (j - i)] = matrix[i * rows + j];
            }

        } else {

            for (int j = i; j < rows; ++j) {
                compressed[i * c + (j - i)] = matrix[i * rows + j];
            }
        }
    }

    return compressed;
}


static vector<double> uncrompress_symmetric_banded_matrix(int rows, int bandwidth, const vector<double> &matrix)
{
    int c = (bandwidth + 1) / 2;
    vector<double> uncompressed (rows * rows, 0);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < c; ++j) {
            int idx = i + j;

            if (idx < rows)
                uncompressed[i * rows + idx] = matrix[i * c + j];
        }
    }

    return uncompressed;
}


static void extract_compressed_lower_upper_diagonal(int rows, int bandwidth, const vector<double> &lower, const vector<double> &upper, vector<float> &compressed_lower, vector<float> &compressed_upper, vector<float> &diagonal)
{
    int columns = lower.size() / rows;
    int c = (bandwidth + 1) / 2;
    // Division by 0 can happen if shift is used
    double eps = numeric_limits<double>::epsilon();

    for (int i = 0; i < rows; ++i) {
        int start = max(i - c + 1, 0);
        for (int j = start; j < start + c - 1; ++j) {
            compressed_lower[i * (c - 1) + j - start] = static_cast<float>(lower[i * columns + j]);
        }
    }

    for (int i = 0; i < rows; ++i) {
        int start = min(i + c - 1, rows - 1);
        for (int j = start; j > i; --j) {
            compressed_upper[i * (c - 1) + c - 2 + j - start] = static_cast<float>(upper[i * columns + j]);
        }
    }

    for (int i = 0; i < rows; ++i) {
        diagonal[i] = static_cast<float>(1.0 / (lower[i * columns + i] + eps));
    }

}


static constexpr double PI = 3.14159265358979323846;


static double sinc(double x)
{
    return x == 0.0 ? 1.0 : sin(x * PI) / (x * PI);
}


static double cube(double x)
{
    return x * x * x;
}


static double calculate_weight(DescaleMode mode, int support, double distance, double b, double c)
{
    distance = abs(distance);

    if (mode == bilinear) {
        return max(1.0 - distance, 0.0);

    } else if (mode == bicubic) {
        if (distance < 1)
            return ((12 - 9 * b - 6 * c) * cube(distance)
                        + (-18 + 12 * b + 6 * c) * (distance*distance) + (6 - 2 * b)) / 6.0;
        else if (distance < 2)
            return ((-b - 6 * c) * cube(distance) + (6 * b+ 30 * c) * (distance*distance)
                        + (-12 * b - 48 * c) * distance + (8 * b + 24 * c)) / 6.0;
        else
            return 0.0;

    } else if (mode == lanczos) {
        return distance < support ? sinc(distance) * sinc(distance / support) : 0.;

    } else if (mode == spline16) {
        if (distance < 1.0) {
            return 1.0 - (1.0 / 5.0 * distance) - (9.0 / 5.0 * (distance*distance)) + cube(distance);
        } else if (distance < 2.0) {
            distance -= 1.0;
            return (-7.0 / 15.0 * distance) + (4.0 / 5.0 * (distance*distance)) - (1.0 / 3.0 * cube(distance));
        } else {
            return 0.0;
        }

    } else if (mode == spline36) {
        if (distance < 1.0) {
            return 1.0 - (3.0 / 209.0 * distance) - (453.0 / 209.0 * (distance*distance)) + (13.0 / 11.0 * cube(distance));
        } else if (distance < 2.0) {
            distance -= 1.0;
            return (-156.0 / 209.0 * distance) + (270.0 / 209.0 * (distance*distance)) - (6.0 / 11.0 * cube(distance));
        } else if (distance < 3.0) {
            distance -= 2.0;
            return (26.0 / 209.0 * distance) - (45.0 / 209.0 * (distance*distance)) + (1.0 / 11.0 * cube(distance));
        } else {
            return 0.0;
        }
    }
}


// Stolen from zimg
static double round_halfup(double x) noexcept
{
    /* When rounding on the pixel grid, the invariant
     *   round(x - 1) == round(x) - 1
     * must be preserved. This precludes the use of modes such as
     * half-to-even and half-away-from-zero.
     */
    bool sign = signbit(x);

    x = round(abs(x));
    return sign ? -x : x;
}


// Most of this is taken from zimg
// https://github.com/sekrit-twc/zimg/blob/ce27c27f2147fbb28e417fbf19a95d3cf5d68f4f/src/zimg/resize/filter.cpp#L227
static vector<double> scaling_weights(DescaleMode mode, int support, int src_dim, int dst_dim, double b, double c, double shift)
{
    double ratio = static_cast<double>(dst_dim) / src_dim;
    vector<double> weights (src_dim * dst_dim, 0);

    for (int i = 0; i < dst_dim; ++i) {

        double total = 0.0;
        double pos = (i + 0.5) / ratio + shift;
        double begin_pos = round_halfup(pos - support) + 0.5;
        for (int j = 0; j < 2 * support; ++j) {
            double xpos = begin_pos + j;
            total += calculate_weight(mode, support, xpos - pos, b, c);
        }
        for (int j = 0; j < 2 * support; ++j) {
            double xpos = begin_pos + j;
            double real_pos;

            // Mirror the position if it goes beyond image bounds.
            if (xpos < 0.0)
                real_pos = -xpos;
            else if (xpos >= src_dim)
                real_pos = min(2.0 * src_dim - xpos, src_dim - 0.5);
            else
                real_pos = xpos;

            auto idx = static_cast<int>(real_pos);
            weights[i * src_dim + idx] += calculate_weight(mode, support, xpos - pos, b, c) / total;
        }
    }

    return weights;
}


// Solve A' A x = A' b for x
static void process_plane_h(int width, int current_height, int &current_width, int bandwidth, const vector<int> &weights_left_idx, const vector<int> &weights_right_idx, const vector<float> &weights,
                            const vector<float> &lower, const vector<float> &upper, const vector<float> &diagonal, const int src_stride, const int dst_stride, const float *srcp, float *dstp)
{
    int c = (bandwidth + 1) / 2;
    int columns = weights.size() / width;
    for (int i = 0; i < current_height; ++i) {

        // Solve LD y = A' b
        for (int j = 0; j < width; ++j) {
            float sum = 0.0;
            int start = max(0, j - c + 1);

            for (int k = weights_left_idx[j]; k < weights_right_idx[j]; ++k)
                sum += weights[j * columns + k - weights_left_idx[j]] * srcp[k];
            dstp[j] = sum;

            sum = 0.0;
            for (int k = start; k < j; ++k) {
                sum += lower[j * (c - 1) + k - start] * dstp[k];
            }

            dstp[j] = (dstp[j] - sum) * diagonal[j];
        }

        // Solve L' x = y
        for (int j = width - 2; j >= 0; --j) {
            float sum = 0.0;
            int start = min(width - 1, j + c - 1);

            for (int k = start; k > j; --k) {
                sum += upper[j * (c - 1) + k - start + c - 2] * dstp[k];
            }

            dstp[j] -= sum;
        }
        srcp += src_stride;
        dstp += dst_stride;
    }
    current_width = width;
}


// Solve A' A x = A' b for x
static void process_plane_v(int height, int current_width, int &current_height, int bandwidth, const vector<int> &weights_left_idx, const vector<int> &weights_right_idx, const vector<float> &weights,
                            const vector<float> &lower, const vector<float> &upper, const vector<float> &diagonal, const int src_stride, const int dst_stride, const float *srcp, float *dstp)
{
    int c = (bandwidth + 1) / 2;
    int columns = weights.size() / height;
    for (int i = 0; i < current_width; ++i) {

        // Solve LD y = A' b
        for (int j = 0; j < height; ++j) {
            float sum = 0.0;
            int start = max(0, j - c + 1);

            for (int k = weights_left_idx[j]; k < weights_right_idx[j]; ++k)
                sum += weights[j * columns + k - weights_left_idx[j]] * srcp[k * src_stride + i];
            dstp[j * dst_stride + i] = sum;

            sum = 0.0;
            for (int k = start; k < j; ++k) {
                sum += lower[j * (c - 1) + k - start] * dstp[k * dst_stride + i];
            }

            dstp[j * dst_stride + i] = (dstp[j * dst_stride + i] - sum) * diagonal[j];
        }

        // Solve L' x = y
        for (int j = height - 2; j >= 0; --j) {
            float sum = 0.0;
            int start = min(height - 1, j + c - 1);

            for (int k = start; k > j; --k) {
                sum += upper[j * (c - 1) + k - start + c - 2] * dstp[k * dst_stride + i];
            }

            dstp[j * dst_stride + i] -= sum;
        }
    }
    current_height = height;
}

static void descale_cleanup(DescaleData *d, const VSAPI *vsapi){
    vsapi->freeNode(d->node);
    d->hk->refcount--;
    d->vk->refcount--;
    if (d->hk->refcount <= 0)
        delete d->hk;
    if (d->vk->refcount <= 0)
        delete d->vk;
    delete d;
}

Matrix* genMatrix(DescaleData *d, DescaleKey * key) {
    // fast-path check if matrix for this key already exists
    cache_lock.lock_shared();
    auto n = cacheMap.find(key);
    cache_lock.unlock_shared();
    if (n != cacheMap.end()) {
        // check the fast-path bool first
        if(!n->second->value->ready) {
            // wait for the matrix to be generated
            n->second->value->lock.lock_shared();
            n->second->value->lock.unlock_shared();
        }
        cacheList.move_page_to_head(n->second);
        return n->second->value;
    }
    // synchronized check if matrix for this key already exists
    cache_lock.lock();
    n = cacheMap.find(key);
    if (n != cacheMap.end()) {
        cache_lock.unlock();
        // check the fast-path bool first
        if(!n->second->value->ready) {
            // wait for the matrix to be generated
            n->second->value->lock.lock_shared();
            n->second->value->lock.unlock_shared();
        }
        cacheList.move_page_to_head(n->second);
        return n->second->value;
    }
    // generate new matrix for given parameters
    auto matrix = new Matrix();
    // lock the matrix-specific mutex
    matrix->lock.lock();
    if(cacheMap.size() > max_cache_size && max_cache_size != -1) {
        // there can only be at least one element in the cacheList
        DescaleKey* k = cacheList.remove_back_page();
        cacheMap.erase(k);
        k->refcount--;
        if(k->refcount <= 0)
            delete k;
    }
    key->refcount++;
    cacheMap[key] = cacheList.add_page_to_head(key, matrix);
    // unlock the global lock so the matrix generation doesn't block the whole filter
    cache_lock.unlock();

    vector<double> weights = scaling_weights(d->mode, d->support, key->dst, key->src, d->b, d->c, key->shift);
    vector<double> transposed_weights = transpose_matrix(key->src, weights);

    matrix->weights_left_idx.resize(key->dst);
    matrix->weights_right_idx.resize(key->dst);

    for (int i = 0; i < key->dst; ++i) {
        for (int j = 0; j < key->src; ++j) {
            if (transposed_weights[i * key->src + j] != 0.0) {
                matrix->weights_left_idx[i] = j;
                break;
            }
        }
        for (int j = key->src - 1; j >= 0; --j) {
            if (transposed_weights[i * key->src + j] != 0.0) {
                matrix->weights_right_idx[i] = j + 1;
                break;
            }
        }
    }

    vector<double> multiplied_weights = multiply_sparse_matrices(key->dst, matrix->weights_left_idx, matrix->weights_right_idx, transposed_weights, weights);

    vector<double> upper (key->dst * key->dst, 0);
    upper = compress_symmetric_banded_matrix(key->dst, d->bandwidth, multiplied_weights);
    banded_ldlt_decomposition(key->dst, d->bandwidth, upper);
    upper = uncrompress_symmetric_banded_matrix(key->dst, d->bandwidth, upper);

    vector<double> lower = transpose_matrix(key->dst, upper);
    multiply_banded_matrix_with_diagonal(key->dst, d->bandwidth, lower);

    transposed_weights = compress_matrix(key->dst, matrix->weights_left_idx, matrix->weights_right_idx, transposed_weights);

    int compressed_columns = transposed_weights.size() / key->dst;
    matrix->weights.resize(key->dst * compressed_columns, 0);
    matrix->diagonal.resize(key->dst, 0);
    matrix->lower.resize(key->dst * ((d->bandwidth + 1) / 2 - 1), 0);
    matrix->upper.resize(key->dst * ((d->bandwidth + 1) / 2 - 1), 0);

    extract_compressed_lower_upper_diagonal(key->dst, d->bandwidth, lower, upper, matrix->lower, matrix->upper, matrix->diagonal);
    for (int i = 0; i < key->dst; ++i) {
        for (int j = 0; j < compressed_columns; ++j) {
            matrix->weights[i * compressed_columns + j] = static_cast<float>(transposed_weights[i * compressed_columns + j]);
        }
    }

    matrix->ready = true;
    matrix->lock.unlock();
    return matrix;
}


static const VSFrameRef *VS_CC descale_get_frame(int n, int activationReason, void **instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi)
{
    auto * d = static_cast<DescaleData *>(*instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);


    } else if (activationReason == arAllFramesReady) {

        const VSFrameRef * src = vsapi->getFrameFilter(n, d->node, frameCtx);
        const VSFormat * fi = d->vi.format;

        int width = vsapi->getFrameWidth(src, 0);
        int height = vsapi->getFrameHeight(src, 0);

        bool process_h = width != d->vi_dst.width;
        bool process_v = height != d->vi_dst.height;

        Matrix * hmatrix;
        Matrix * vmatrix;

        if (process_h) {
            hmatrix = genMatrix(d, d->hk);
        }
        if (process_v) {
            vmatrix = genMatrix(d, d->vk);
        }

        VSFrameRef * intermediate = vsapi->newVideoFrame(fi, d->vi_dst.width, d->vi.height, nullptr, core);
        VSFrameRef * dst = vsapi->newVideoFrame(fi, d->vi_dst.width, d->vi_dst.height, src, core);

        for (int plane = 0; plane < d->vi.format->numPlanes; ++plane) {
            int cur_width = width;
            int cur_height = height;

            const int src_stride = vsapi->getStride(src, plane) / sizeof(float);
            const auto * srcp = reinterpret_cast<const float *>(vsapi->getReadPtr(src, plane));

            if (process_h && process_v) {
                const int intermediate_stride = vsapi->getStride(intermediate, plane) / sizeof(float);
                auto * VS_RESTRICT intermediatep = reinterpret_cast<float *>(vsapi->getWritePtr(intermediate, plane));

                process_plane_h(d->vi_dst.width, cur_height, cur_width, d->bandwidth, hmatrix->weights_left_idx,
                                hmatrix->weights_right_idx, hmatrix->weights, hmatrix->lower, hmatrix->upper,
                                hmatrix->diagonal, src_stride, intermediate_stride, srcp, intermediatep);

                const int dst_stride = vsapi->getStride(dst, plane) / sizeof(float);
                auto * VS_RESTRICT dstp = reinterpret_cast<float *>(vsapi->getWritePtr(dst, plane));

                process_plane_v(d->vi_dst.height, cur_width, cur_height, d->bandwidth, vmatrix->weights_left_idx,
                                vmatrix->weights_right_idx, vmatrix->weights, vmatrix->lower, vmatrix->upper,
                                vmatrix->diagonal, intermediate_stride, dst_stride, intermediatep, dstp);

            } else if (process_h) {
                const int dst_stride = vsapi->getStride(dst, plane) / sizeof(float);
                auto * VS_RESTRICT dstp = reinterpret_cast<float *>(vsapi->getWritePtr(dst, plane));

                process_plane_h(d->vi_dst.width, cur_height, cur_width, d->bandwidth, hmatrix->weights_left_idx,
                                hmatrix->weights_right_idx, hmatrix->weights, hmatrix->lower, hmatrix->upper,
                                hmatrix->diagonal, src_stride, dst_stride, srcp, dstp);

            } else if (process_v) {
                const int dst_stride = vsapi->getStride(dst, plane) / sizeof(float);
                auto * VS_RESTRICT dstp = reinterpret_cast<float *>(vsapi->getWritePtr(dst, plane));

                process_plane_v(d->vi_dst.height, cur_width, cur_height, d->bandwidth, vmatrix->weights_left_idx,
                                vmatrix->weights_right_idx, vmatrix->weights, vmatrix->lower, vmatrix->upper,
                                vmatrix->diagonal, src_stride, dst_stride, srcp, dstp);
            }
        }

        vsapi->freeFrame(intermediate);

        if (process_h || process_v) {
            vsapi->freeFrame(src);

            return dst;

        } else {
            vsapi->freeFrame(dst);

            return src;
        }
    }

    return nullptr;
}

static void VS_CC descale_init(VSMap *in, VSMap *out, void **instanceData, VSNode *node, VSCore *core, const VSAPI *vsapi)
{
    auto * d = static_cast<DescaleData *>(*instanceData);
    vsapi->setVideoInfo(&d->vi_dst, 1, node);
}

static void VS_CC descale_free(void *instanceData, VSCore *core, const VSAPI *vsapi)
{
    descale_cleanup(static_cast<DescaleData *>(instanceData), vsapi);
}

static void VS_CC descale_change_cache_size(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi)
{
    int err;
    auto size = static_cast<int>(vsapi->propGetInt(in, "size", 0, &err));
    if(err){
        vsapi->setError(out, "Descale: Invalid cache size.");
        return;
    }
    if (size == 0 )
        size = 1;
    lock_guard<shared_mutex>lock(cache_lock);
    max_cache_size = size;
    if (size <= -1)
        return;
    while(cacheMap.size() > size) {
        DescaleKey * k = cacheList.remove_back_page();
        cacheMap.erase(k);
    }
}


static void VS_CC descale_create(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi)
{
    auto mode = static_cast<DescaleMode>(reinterpret_cast<uintptr_t>(userData));

    auto d = new DescaleData();

    d->mode = mode;
    d->node = vsapi->propGetNode(in, "src", 0, nullptr);
    d->vi = *vsapi->getVideoInfo(d->node);
    d->vi_dst = *vsapi->getVideoInfo(d->node);
    int err;

    if (!isConstantFormat(&d->vi) || (d->vi.format->id != pfGrayS && d->vi.format->id != pfRGBS && d->vi.format->id != pfYUV444PS)) {
        vsapi->setError(out, "Descale: Constant format GrayS, RGBS, and YUV444PS are the only supported input formats.");
        descale_cleanup(d, vsapi);
        return;
    }

    d->vi_dst.width = static_cast<int>(vsapi->propGetInt(in, "width", 0, nullptr));
    d->vi_dst.height = static_cast<int>(vsapi->propGetInt(in, "height", 0, nullptr));

    if (d->vi_dst.width < 1 || d->vi_dst.height < 1) {
        vsapi->setError(out, "Descale: width and height must be bigger than 0.");
        descale_cleanup(d, vsapi);
        return;
    }

    if (d->vi_dst.width > d->vi.width || d->vi_dst.height > d->vi.height) {
        vsapi->setError(out, "Descale: Output dimension has to be smaller than or equal to input dimension.");
        descale_cleanup(d, vsapi);
        return;
    }

    d->shift_h = static_cast<float>(vsapi->propGetFloat(in, "src_left", 0, &err));
    if (err)
        d->shift_h = 0;

    d->shift_v = static_cast<float>(vsapi->propGetFloat(in, "src_top", 0, &err));
    if (err)
        d->shift_v = 0;

    string funcname;

    if (mode == bilinear) {
        d->support = 1;
        funcname = "Debilinear";

    } else if (mode == bicubic) {
        d->b = vsapi->propGetFloat(in, "b", 0, &err);
        if (err)
            d->b = 1. / 3.;

        d->c = vsapi->propGetFloat(in, "c", 0, &err);
        if (err)
            d->c = 1. / 3.;

        d->support = 3;
        funcname = "Debicubic";

    } else if (mode == lanczos) {
        d->taps = static_cast<int>(vsapi->propGetInt(in, "taps", 0, &err));
        if (err)
            d->taps = 3;

        if (d->taps < 1) {
            vsapi->setError(out, "Descale: taps must be bigger than 0.");
            descale_cleanup(d, vsapi);
            return;
        }
        d->support = d->taps;
        funcname = "Delanczos";

    } else if (mode == spline16) {
        d->support = 2;
        funcname = "Despline16";

    } else {
        //  if (mode == spline36)
        d->support = 3;
        funcname = "Despline36";
    }

    d->bandwidth = d->support * 4 - 1;
    d->init_keys();

    vsapi->createFilter(in, out, funcname.c_str(), descale_init, descale_get_frame, descale_free, fmParallel, 0, d, core);
}


VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin)
{
    configFunc("toggaf.asi.xe", "descale", "Undo linear interpolation", VAPOURSYNTH_API_VERSION, 1, plugin);

    registerFunc("Debilinear",
            "src:clip;"
            "width:int;"
            "height:int;"
            "src_left:float:opt;"
            "src_top:float:opt;",
            descale_create, reinterpret_cast<void *>(bilinear), plugin);

    registerFunc("Debicubic",
            "src:clip;"
            "width:int;"
            "height:int;"
            "b:float:opt;"
            "c:float:opt;"
            "src_left:float:opt;"
            "src_top:float:opt;",
            descale_create, reinterpret_cast<void *>(bicubic), plugin);

    registerFunc("Delanczos",
            "src:clip;"
            "width:int;"
            "height:int;"
            "taps:int:opt;"
            "src_left:float:opt;"
            "src_top:float:opt;",
            descale_create, reinterpret_cast<void *>(lanczos), plugin);

    registerFunc("Despline16",
            "src:clip;"
            "width:int;"
            "height:int;"
            "src_left:float:opt;"
            "src_top:float:opt;",
            descale_create, reinterpret_cast<void *>(spline16), plugin);

    registerFunc("Despline36",
            "src:clip;"
            "width:int;"
            "height:int;"
            "src_left:float:opt;"
            "src_top:float:opt;",
            descale_create, reinterpret_cast<void *>(spline36), plugin);

    registerFunc("CacheSize",
             "size:int;",
             descale_change_cache_size, nullptr, plugin);
}
