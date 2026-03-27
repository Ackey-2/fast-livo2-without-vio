/**
 * voxel_downsample.h - 手写体素下采样
 *
 * 替代 PCL 的 VoxelGrid 滤波器，避免潜在的库 bug。
 *
 * 使用: #include "voxel_downsample.h"
 */

#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <unordered_map>
#include <Eigen/Dense>

/**
 * 体素坐标键值，用于 unordered_map
 *
 * 原理: 将连续的 3D 空间离散化为固定大小的格子。
 *       每个点按 floor(x/leaf_size) 计算所属格子的整数坐标 (ix, iy, iz)。
 *       同一格子内的所有点取均值，输出一个代表点。
 */
struct VoxelKey {
    int ix, iy, iz;

    bool operator==(const VoxelKey &other) const {
        return ix == other.ix && iy == other.iy && iz == other.iz;
    }
};

// VoxelKey 的哈希函数
struct VoxelKeyHash {
    std::size_t operator()(const VoxelKey &k) const {
        // 用三个大质数做混合，减少哈希碰撞
        std::size_t h = 0;
        h ^= std::hash<int>()(k.ix) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>()(k.iy) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>()(k.iz) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

/**
 * 每个体素格子内的累积数据
 */
struct VoxelAccum {
    double sx = 0, sy = 0, sz = 0;  // 坐标累加
    double si = 0;                   // intensity 累加
    int count = 0;                   // 点数
};

/**
 * voxel_downsample - 体素下采样
 *
 * @param input      输入点云
 * @param output     输出点云（会被清空后写入）
 * @param leaf_size  体素边长（米）
 *
 * 算法:
 *   1. 对每个点，计算它所属体素的整数坐标 (ix, iy, iz)
 *   2. 将同一体素内所有点的坐标和 intensity 累加
 *   3. 最后对每个体素输出一个均值点
 *
 * 时间复杂度: O(N)，N 为输入点数
 * 空间复杂度: O(K)，K 为非空体素数
 */
inline void voxel_downsample(
    const pcl::PointCloud<pcl::PointXYZI> &input,
    pcl::PointCloud<pcl::PointXYZI> &output,
    double leaf_size)
{
    output.clear();

    if (input.empty() || leaf_size <= 0) return;

    double inv_leaf = 1.0 / leaf_size;

    // 预估哈希表大小，减少 rehash 开销
    std::unordered_map<VoxelKey, VoxelAccum, VoxelKeyHash> voxels;
    voxels.reserve(input.size() / 4);

    // ---- 第一遍: 按体素分组，累积坐标 ----
    for (const auto &pt : input.points) {
        // 跳过 NaN 点
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z))
            continue;

        VoxelKey key;
        key.ix = static_cast<int>(std::floor(pt.x * inv_leaf));
        key.iy = static_cast<int>(std::floor(pt.y * inv_leaf));
        key.iz = static_cast<int>(std::floor(pt.z * inv_leaf));

        auto &acc = voxels[key];
        acc.sx += pt.x;
        acc.sy += pt.y;
        acc.sz += pt.z;
        acc.si += pt.intensity;
        acc.count++;
    }

    // ---- 第二遍: 每个体素输出一个均值点 ----
    output.reserve(voxels.size());

    for (const auto &pair : voxels) {
        const VoxelAccum &acc = pair.second;
        double inv_n = 1.0 / acc.count;

        pcl::PointXYZI pt;
        pt.x = acc.sx * inv_n;
        pt.y = acc.sy * inv_n;
        pt.z = acc.sz * inv_n;
        pt.intensity = acc.si * inv_n;
        output.push_back(pt);
    }
}