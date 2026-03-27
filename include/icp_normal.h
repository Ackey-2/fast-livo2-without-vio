/**
 * ============================================================================
 * icp_normal.h - 基于法向量的点到平面 ICP 验证
 *
 * 使用方法: 在 LIVMapper.cpp 顶部 #include "icp_normal.h"
 *
 * 依赖: Eigen, PCL
 * ============================================================================
 */

#pragma once

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <vector>
#include <cmath>

/**
 * hat() - 向量的反对称矩阵（skew-symmetric matrix）
 *
 * 将 3D 向量 v = [vx, vy, vz] 转换为 3x3 反对称矩阵:
 *
 *       [  0  -vz   vy ]
 *   v^ = [ vz   0  -vx ]
 *       [-vy  vx    0  ]
 *
 * 物理意义: 叉积 a × b 可以写成矩阵乘法 hat(a) * b
 * 在 ICP 中用于构建旋转扰动的雅可比矩阵
 */
inline Eigen::Matrix3d hat(const Eigen::Vector3d &v)
{
    Eigen::Matrix3d m;
    m <<     0, -v[2],  v[1],
          v[2],     0, -v[0],
         -v[1],  v[0],     0;
    return m;
}

/**
 * Exp() - SO(3) 指数映射: 旋转向量 → 旋转矩阵
 *
 * 给定一个旋转向量 omega (方向 = 旋转轴, 模长 = 旋转角度/弧度)，
 * 返回对应的 3x3 旋转矩阵。
 *
 * 使用 Rodrigues 公式:
 *   R = I + sin(θ)/θ * [ω]× + (1 - cos(θ))/θ² * [ω]×²
 *
 * 当角度很小时（θ → 0），用泰勒展开近似以避免数值除零。
 */
inline Eigen::Matrix3d Exp(const Eigen::Vector3d &omega)
{
    double theta = omega.norm();

    if (theta < 1e-10) {
        // 小角度近似: R ≈ I + [ω]×
        return Eigen::Matrix3d::Identity() + hat(omega);
    }

    Eigen::Matrix3d omega_hat = hat(omega);

    // Rodrigues 公式
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity()
                        + (std::sin(theta) / theta) * omega_hat
                        + ((1.0 - std::cos(theta)) / (theta * theta)) * omega_hat * omega_hat;
    return R;
}

/**
 * icp_normal() - 基于法向量的点到平面 ICP 精确配准
 *
 * 这是回环检测中的关键验证步骤。STD 描述子给出了一个粗略的相对位姿估计，
 * 但可能不够精确。ICP 用两帧点云的实际几何形状来精化这个变换，
 * 并通过收敛性和法向量约束的充分性来判断这个回环是否可靠。
 *
 * ===================== 算法原理 =====================
 *
 * 点到平面 ICP 的目标: 找到旋转 R 和平移 t，使得对于源点云中的每个点 pi，
 * 它变换后与目标点云中最近邻 qi 之间的"点到平面距离"最小:
 *
 *   minimize Σ |ni^T * (R*pi + t - qi)|²
 *
 * 其中 ni 是目标点 qi 处的法向量。
 *
 * 为什么用"点到平面"而不是"点到点"？
 *   在平坦区域，点到点 ICP 会沿着平面方向产生虚假的位移；
 *   点到平面 ICP 只约束法线方向的距离，允许沿平面方向自由滑动，
 *   这更符合物理意义，收敛也更快。
 *
 * ===================== 匹配过滤条件 =====================
 *
 * 不是所有最近邻对都参与优化，需要满足三个条件:
 *   1. 法向量夹角检查: |ni_src - ni_tar| < 0.2 或 |ni_src + ni_tar| < 0.2
 *      → 两个法向量方向一致（或反向，因为法向量可能方向不定）
 *   2. 点到平面距离 < 0.5m → 点确实靠近对应平面
 *   3. 点到点距离 < 3.0m → 避免太远的错误匹配
 *
 *   收敛后会收紧条件 (0.1, 0.1, 0.1, 1.0) 再迭代一轮以提高精度。
 *
 * ===================== 收敛判定与质量评估 =====================
 *
 * 收敛条件: 旋转增量 < 1e-3 rad 且 平移增量 < 1e-3 m
 *
 * 质量评估（关键）: 用法向量的外积矩阵 Σ(ni*ni^T) 的最小特征值来衡量
 * 约束的"充分性"：
 *   - 如果所有匹配的法向量都指向同一个方向（比如都在一面墙上），
 *     那只有一个方向有约束，另外两个方向是自由的 → 最小特征值 ≈ 0 → 不可靠
 *   - 如果法向量分布在三个不同方向（比如地面+两面墙），
 *     三个方向都有约束 → 最小特征值 > 阈值 → 可靠
 *   - icp_eigval 阈值（默认 14）就是这个最小特征值的门槛
 *
 * 返回 true: ICP 收敛 且 法向量约束充分（最小特征值 > icp_eigval）
 *
 * ===================== 参数说明 =====================
 *
 * @param pl_src    源点云（当前关键帧的平面点云，局部坐标系）
 *                  类型需要有 normal_x/y/z 字段（如 PointXYZINormal）
 * @param pl_tar    目标点云（匹配到的历史关键帧的平面点云，局部坐标系）
 * @param pose      [输入/输出] 相对位姿变换
 *                  .first  = 平移向量 t（源→目标）
 *                  .second = 旋转矩阵 R（源→目标）
 *                  输入时是 STD 给的粗估计，输出时被 ICP 精化
 * @param icp_eigval 法向量约束矩阵的最小特征值阈值（默认 14）
 *
 * @return true = ICP 验证通过（收敛 + 约束充分），false = 验证失败
 */
inline bool icp_normal(
    pcl::PointCloud<pcl::PointXYZINormal> &pl_src,    // 源点云（当前帧）
    pcl::PointCloud<pcl::PointXYZINormal> &pl_tar,    // 目标点云（匹配帧）
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> &pose, // 相对变换（被精化）
    double icp_eigval)                                  // 特征值阈值
{
    // ---- 1. 用目标点云构建 KD-Tree，用于最近邻搜索 ----

    pcl::KdTreeFLANN<pcl::PointXYZ> kd_tree;
    pcl::PointCloud<pcl::PointXYZ> input_cloud;

    for (auto &ap : pl_tar.points)
    {
        pcl::PointXYZ pi;
        pi.x = ap.x; pi.y = ap.y; pi.z = ap.z;
        input_cloud.push_back(pi);
    }
    kd_tree.setInputCloud(input_cloud.makeShared());

    // 最近邻搜索的输出缓冲区（每次只查 1 个最近邻）
    std::vector<int>   pointIdxNKNSearch(1);
    std::vector<float> pointNKNSquaredDistance(1);

    // ---- 2. 匹配过滤参数 ----
    // paras = [法向量差阈值, 法向量和阈值, 点到平面距离阈值, 点到点距离阈值]
    // 先用宽松参数迭代，收敛后切换为严格参数再迭代
    Eigen::Vector4d paras(0.2, 0.2, 0.5, 3);
    int is_converge = 0;  // 是否已经收敛过一次

    int ssize = pl_src.size();
    int match_num = 0;
    Eigen::Matrix3d mat_norm;  // 法向量外积矩阵 Σ(ni*ni^T)，用于评估约束充分性

    // ---- 3. ICP 迭代 ----
    for (int iterCount = 0; iterCount < 20; iterCount++)
    {
        // 海森矩阵和雅可比向量（高斯-牛顿法求解 6DoF 位姿增量）
        Eigen::Matrix<double, 6, 6> Hess; Hess.setZero();
        Eigen::Matrix<double, 6, 1> JacT; JacT.setZero();
        double resi = 0;      // 总残差
        match_num = 0;         // 有效匹配数
        mat_norm.setZero();

        for (int i = 0; i < ssize; i++)
        {
            auto &searchPoint = pl_src[i];

            // ---- 3a. 将源点从局部坐标系变换到目标坐标系 ----
            Eigen::Vector3d plocal(searchPoint.x, searchPoint.y, searchPoint.z);
            Eigen::Vector3d pi = pose.second * plocal + pose.first;
            // pi = R * plocal + t

            pcl::PointXYZ use_search_point;
            use_search_point.x = pi[0];
            use_search_point.y = pi[1];
            use_search_point.z = pi[2];

            // 源点的法向量也要旋转到目标坐标系
            Eigen::Vector3d ni(searchPoint.normal_x, searchPoint.normal_y, searchPoint.normal_z);
            ni = pose.second * ni;

            // ---- 3b. 在目标点云中找最近邻 ----
            if (kd_tree.nearestKSearch(use_search_point, 1,
                                       pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
            {
                auto &nearstPoint = pl_tar[pointIdxNKNSearch[0]];
                Eigen::Vector3d tpi(nearstPoint.x, nearstPoint.y, nearstPoint.z);
                Eigen::Vector3d tni(nearstPoint.normal_x, nearstPoint.normal_y, nearstPoint.normal_z);

                // ---- 3c. 匹配对过滤 ----
                Eigen::Vector3d normal_inc = ni - tni;  // 法向量之差
                Eigen::Vector3d normal_add = ni + tni;  // 法向量之和（处理法向量反向的情况）
                double point_to_point_dis = (pi - tpi).norm();
                double point_to_plane = fabs(tni.transpose() * (pi - tpi));

                // 条件:
                //   (法向量几乎同向 或 法向量几乎反向) 且
                //   点到平面距离够小 且
                //   点到点距离够小
                if ((normal_inc.norm() < paras[0] || normal_add.norm() < paras[1])
                    && point_to_plane < paras[2]
                    && point_to_point_dis < paras[3])
                {
                    // ---- 3d. 计算残差和雅可比 ----
                    // 残差: r = ni^T * (R*plocal + t - qi)  （点到平面距离，带符号）
                    double rr = tni.dot(pi - tpi);

                    // 雅可比 J = ∂r/∂(δω, δt)
                    //   对旋转的偏导: hat(plocal) * R^T * tni
                    //     推导: 旋转扰动 δR ≈ I + [δω]×
                    //           r = tni^T * ((I+[δω]×)*R*plocal + t - qi)
                    //           ∂r/∂δω = tni^T * [R*plocal]× 的转置形式
                    //   对平移的偏导: tni
                    //     推导: r = tni^T * (R*plocal + t + δt - qi)
                    //           ∂r/∂δt = tni
                    Eigen::Matrix<double, 6, 1> jac;
                    jac.head(3) = hat(plocal) * pose.second.transpose() * tni;
                    jac.tail(3) = tni;

                    // 高斯-牛顿: H += J*J^T, b += J*r
                    Hess += jac * jac.transpose();
                    JacT += jac * rr;
                    resi += 0.5 * rr * rr;
                    match_num++;

                    // 累积法向量外积矩阵（用于最终质量评估）
                    mat_norm += tni * tni.transpose();
                }
            }
        }

        // ---- 3e. 求解位姿增量 ----
        // H * δx = -b  →  δx = H^{-1} * (-b)
        // δx = [δω(3), δt(3)]
        Eigen::Matrix<double, 6, 1> dxi = Hess.ldlt().solve(-JacT);

        // 更新位姿
        pose.second = pose.second * Exp(dxi.head(3));  // R = R * Exp(δω)
        pose.first  = pose.first  + dxi.tail(3);       // t = t + δt

        // ---- 3f. 收敛检查 ----
        if (dxi.head(3).norm() < 1e-3 && dxi.tail(3).norm() < 1e-3)
        {
            if (is_converge)
            {
                // 第二次收敛（严格参数下也收敛了），可以退出
                break;
            }
            else
            {
                // 第一次收敛，切换为更严格的匹配参数，再迭代一轮
                // 这样可以剔除之前宽松参数下的错误匹配，进一步提高精度
                paras << 0.1, 0.1, 0.1, 1;
                is_converge = 1;
            }
        }
    }

    // ---- 4. 质量评估 ----
    // mat_norm = Σ(ni * ni^T) 是法向量的协方差矩阵
    // 它的特征值反映了法向量在各方向上的分布:
    //   三个特征值都大 → 法向量分布均匀（三个方向都有约束）→ 好
    //   最小特征值 ≈ 0 → 某个方向缺乏约束 → 位姿在该方向不可靠
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(mat_norm);
    Eigen::Vector3d eig_vec = saes.eigenvalues();
    printf("eigvalue: %lf %lf %lf %d\n", eig_vec[0], eig_vec[1], eig_vec[2], is_converge);

    // 要求: 最小特征值 > 阈值 且 已经收敛
    return eig_vec[0] > icp_eigval && is_converge == 1;
}