/**
 * loop_detector.h  v4
 * 
 * 修复：
 *   1. ★关键 Bug★：keyframe_infos_ 里 scan_id 存的是 LIO 帧号(如2404)，
 *      但 SearchLoop 返回的 result.first 是 BTC 顺序编号(如44)，
 *      导致漂移率检查永远找不到匹配帧 → 真回环全被错杀
 *   2. 加全链路诊断日志，每一关都打印结果
 */

#ifndef LOOP_DETECTOR_H
#define LOOP_DETECTOR_H

#include "BTC.h"
#include <deque>
#include <unordered_map>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>

// ===== 手写体素降采样 =====
namespace loop_utils {
struct VK {
  int x, y, z;
  bool operator==(const VK &o) const { return x==o.x && y==o.y && z==o.z; }
};
struct VH {
  size_t operator()(const VK &k) const {
    return ((size_t)k.x*73856093)^((size_t)k.y*19349663)^((size_t)k.z*83492791);
  }
};
inline void downsample(const pcl::PointCloud<pcl::PointXYZI>::Ptr &in,
                       pcl::PointCloud<pcl::PointXYZI>::Ptr &out, float leaf) {
  out->clear();
  if (!in || in->empty() || leaf < 0.001f) { if(in) *out=*in; return; }
  float inv = 1.0f / leaf;
  struct A { double x=0,y=0,z=0,i=0; int n=0; };
  std::unordered_map<VK,A,VH> m; m.reserve(in->size()/4);
  for (auto &p : in->points) {
    VK k{(int)std::floor(p.x*inv),(int)std::floor(p.y*inv),(int)std::floor(p.z*inv)};
    auto &a=m[k]; a.x+=p.x; a.y+=p.y; a.z+=p.z; a.i+=p.intensity; a.n++;
  }
  out->reserve(m.size());
  for (auto &kv : m) {
    auto &a=kv.second; float r=1.0f/a.n;
    pcl::PointXYZI p; p.x=a.x*r; p.y=a.y*r; p.z=a.z*r; p.intensity=a.i*r;
    out->push_back(p);
  }
}
} // namespace loop_utils

// ===== 配置 =====
struct LoopDetectorConfig
{
  int    win_size        = 10;
  double score_thresh    = 0.10;   // BTC 粗筛（放低，靠 ICP 把关）
  bool   is_high_fly     = false;
  double ds_size         = 0.15;
  double min_key_dist    = 1.0;
  double btc_voxel_size  = 0.3;

  // ICP 验证参数
  double icp_eigval_thresh = 1.0;    // 仅 BTC score<0.5 时要求特征值（降到1.0）
  double drift_ratio_thresh = 0.05;  // 漂移率上限
  int    cooldown_frames   = 30;     // 回环冷却
};

// ===== 回环检测器 =====
class LoopDetector
{
public:
  LoopDetector() : std_manager_(nullptr), frame_count_(0), keyframe_count_(0), cooldown_(0) {}
  ~LoopDetector() { delete std_manager_; }

  void init(const LoopDetectorConfig& cfg = LoopDetectorConfig())
  {
    cfg_ = cfg;
    ConfigSetting btc_cfg;
    init_config_setting(btc_cfg, cfg.is_high_fly ? 1 : 0);

    btc_cfg.voxel_size_     = cfg.btc_voxel_size;
    btc_cfg.voxel_init_num_ = 5;
    btc_cfg.summary_min_thre_ = 2;
    btc_cfg.proj_plane_num_   = 3;
    btc_cfg.proj_dis_max_     = 5;
    btc_cfg.proj_image_high_inc_ = 0.3;
    btc_cfg.descriptor_min_len_  = 0.5;
    btc_cfg.non_max_suppression_radius_ = 1.0;
    btc_cfg.line_filter_enable_ = 0;
    btc_cfg.normal_threshold_ = 0.4;
    btc_cfg.dis_threshold_    = 1.0;
    btc_cfg.icp_threshold_    = 0.10;

    std_manager_ = new STDescManager(btc_cfg);
    printf("\033[1;36m[LoopDetector] Init OK. win=%d btc>=%.2f icp_eig>=%.1f drift<=%.2f cool=%d\033[0m\n",
           cfg_.win_size, cfg_.score_thresh, cfg_.icp_eigval_thresh,
           cfg_.drift_ratio_thresh, cfg_.cooldown_frames);
  }

  void addScanAndDetect(
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_world,
    const Eigen::Matrix3d& R,
    const Eigen::Vector3d& p,
    int frame_id)
  {
    if (!std_manager_ || !cloud_world || cloud_world->empty()) return;

    FrameData fd;
    fd.cloud = cloud_world; fd.R = R; fd.p = p; fd.id = frame_id;
    frame_buffer_.push_back(fd);
    frame_count_++;

    if ((int)frame_buffer_.size() < cfg_.win_size) return;

    Eigen::Vector3d cur_p = frame_buffer_.back().p;
    if (keyframe_count_ > 0) {
      double step = (cur_p - last_key_pos_).norm();
      if (step < cfg_.min_key_dist) { frame_buffer_.pop_front(); return; }
      total_journey_ += step;
    }
    last_key_pos_ = cur_p;

    // ---- 合并关键帧 ----
    pcl::PointCloud<pcl::PointXYZI>::Ptr merged(new pcl::PointCloud<pcl::PointXYZI>());
    Eigen::Matrix3d ref_R = frame_buffer_.back().R;
    Eigen::Vector3d ref_p = frame_buffer_.back().p;
    int ref_id = frame_buffer_.back().id;

    for (int i = 0; i < cfg_.win_size; i++) {
      auto& f = frame_buffer_[i];
      for (auto& pt : f.cloud->points) {
        Eigen::Vector3d v_local = ref_R.transpose() * (Eigen::Vector3d(pt.x,pt.y,pt.z) - ref_p);
        pcl::PointXYZI pi;
        pi.x=v_local[0]; pi.y=v_local[1]; pi.z=v_local[2]; pi.intensity=pt.intensity;
        merged->push_back(pi);
      }
    }
    for (int i = 0; i < cfg_.win_size; i++) frame_buffer_.pop_front();

    if (cfg_.ds_size > 0.001 && merged->size() > 100) {
      pcl::PointCloud<pcl::PointXYZI>::Ptr ds(new pcl::PointCloud<pcl::PointXYZI>());
      loop_utils::downsample(merged, ds, cfg_.ds_size);
      merged = ds;
    }

    // ★★★ Bug 修复：scan_id 用 keyframe_count_（= BTC 顺序编号），不用 ref_id ★★★
    KeyframeInfo ki;
    ki.pos = ref_p;
    ki.journey = total_journey_;
    ki.btc_id = keyframe_count_;   // ← BTC 顺序编号（和 SearchLoop 返回值匹配）
    ki.lio_frame_id = ref_id;       // ← LIO 帧号（仅用于打印）
    keyframe_infos_.push_back(ki);

    printf("[LoopDetector] KF#%d (frame=%d): %zu pts, journey=%.1fm\n",
           keyframe_count_, ref_id, merged->size(), total_journey_);

    if (merged->size() < 100) { keyframe_count_++; return; }

    // ---- 生成 BTC 描述子 ----
    std::vector<STD> stds_vec;
    std_manager_->GenerateSTDescs(merged, stds_vec, ref_id);

    size_t n_planes = std_manager_->plane_cloud_vec_.back() ?
                      std_manager_->plane_cloud_vec_.back()->size() : 0;
    printf("[LoopDetector] planes=%zu, STDs=%zu\n", n_planes, stds_vec.size());

    cooldown_--;

    // ---- 搜索回环 ----
    if (keyframe_count_ > 0 && stds_vec.size() > 0 && cooldown_ <= 0)
    {
      std::pair<int, double> result(-1, 0);
      std::pair<Eigen::Vector3d, Eigen::Matrix3d> tf;
      std::vector<std::pair<STD, STD>> pairs;

      std_manager_->SearchLoop(stds_vec, result, tf, pairs,
                                std_manager_->plane_cloud_vec_.back());

      if (result.first >= 0)
      {
        // ========== 第一关：BTC 粗匹配 ==========
        printf("\033[1;33m[Loop] Gate1 BTC: cur_kf=%d(frame=%d) → match_kf=%d, score=%.4f (need>%.2f) %s\033[0m\n",
               keyframe_count_, ref_id, result.first, result.second, cfg_.score_thresh,
               result.second > cfg_.score_thresh ? "PASS" : "FAIL");

        if (result.second > cfg_.score_thresh)
        {
          // ========== 第二关：ICP 精化变换 ==========
          // 注意：plane_cloud 只有 30-50 个点，不能要求太高
          // VoxelSLAM 有几百个平面所以 eigval>10 合理，我们只要求收敛即可
          double icp_min_eig = 0;
          bool icp_converge = false;
          icpRefine(
            *(std_manager_->plane_cloud_vec_.back()),
            *(std_manager_->plane_cloud_vec_[result.first]),
            tf, icp_min_eig, icp_converge);

          // ICP 判定策略（参考 VoxelSLAM）：
          //   - BTC score 很高(>0.5)：ICP 只需收敛
          //   - BTC score 一般(0.1~0.5)：ICP 需要收敛 + 最小特征值 > 阈值
          bool icp_pass = false;
          if (result.second > 0.5 && icp_converge) {
            icp_pass = true;  // 高置信 BTC，ICP 收敛即可
            printf("\033[1;33m[Loop] Gate2 ICP: PASS (high BTC + converge)\033[0m\n");
          } else if (icp_converge && icp_min_eig > cfg_.icp_eigval_thresh) {
            icp_pass = true;  // 低置信 BTC，需要 ICP 特征值兜底
            printf("\033[1;33m[Loop] Gate2 ICP: PASS (eig=%.2f>%.1f)\033[0m\n",
                   icp_min_eig, cfg_.icp_eigval_thresh);
          } else {
            printf("\033[1;31m[Loop] Gate2 ICP: FAIL (converge=%d eig=%.2f)\033[0m\n",
                   icp_converge, icp_min_eig);
          }

          if (icp_pass)
          {
            // ========== 第三关：漂移率检查 ==========
            // ★ 用 btc_id 查找（不再用 lio_frame_id）★
            double match_journey = 0;
            Eigen::Vector3d match_pos = Eigen::Vector3d::Zero();
            int match_lio_frame = -1;
            bool found = false;

            for (auto& info : keyframe_infos_) {
              if (info.btc_id == result.first) {
                match_journey = info.journey;
                match_pos = info.pos;
                match_lio_frame = info.lio_frame_id;
                found = true;
                break;
              }
            }

            if (!found) {
              printf("\033[1;31m[Loop] Gate3: Cannot find KF btc_id=%d in history!\033[0m\n", result.first);
            }
            else
            {
              double span = total_journey_ - match_journey;
              double drift = (cur_p - match_pos).norm();
              double ratio = (span > 1.0) ? (drift / span) : 1.0;

              printf("\033[1;33m[Loop] Gate3 Drift: drift=%.2fm span=%.1fm ratio=%.4f (need<%.4f) %s\033[0m\n",
                     drift, span, ratio, cfg_.drift_ratio_thresh,
                     ratio < cfg_.drift_ratio_thresh ? "PASS" : "FAIL");

              if (ratio < cfg_.drift_ratio_thresh)
              {
                printf("\033[1;32m╔══════════════════════════════════════════════════╗\033[0m\n");
                printf("\033[1;32m║          LOOP CLOSURE CONFIRMED!                 ║\033[0m\n");
                printf("\033[1;32m║  cur: KF#%d (frame %d)                           ║\033[0m\n", keyframe_count_, ref_id);
                printf("\033[1;32m║  match: KF#%d (frame %d)                         ║\033[0m\n", result.first, match_lio_frame);
                printf("\033[1;32m║  BTC=%.4f  drift=%.2fm  span=%.1fm              ║\033[0m\n", result.second, drift, span);
                printf("\033[1;32m║  t=(%.2f, %.2f, %.2f)                            ║\033[0m\n",
                       tf.first[0], tf.first[1], tf.first[2]);
                printf("\033[1;32m╚══════════════════════════════════════════════════╝\033[0m\n");
                cooldown_ = cfg_.cooldown_frames;
              }
            }
          }
          else  // ICP failed
          {
            // 不做任何事，已经打印了 FAIL
          }
        }
      }
      else if (result.first >= 0)
      {
        // BTC 找到候选但分数不够（仅在有候选时打印，避免刷屏）
        // 已经在上面打印了 FAIL
      }
    }

    // ---- 数据库统计 ----
    {
      size_t total_stds = 0;
      size_t db_mem = 0;
      for (auto &kv : std_manager_->data_base_) {
        total_stds += kv.second.size();
        db_mem += kv.second.size() * 280 + sizeof(STD_LOC);
      }
      size_t plane_mem = 0, total_planes = 0;
      for (auto &pc : std_manager_->plane_cloud_vec_) {
        if (pc) { total_planes += pc->size(); plane_mem += pc->size() * sizeof(pcl::PointXYZINormal); }
      }
      printf("\033[1;35m[LoopDB] kf:%d | db:%zu/%zu (%.1fMB) | planes:%zu/%zu (%.1fMB) | total:%.1fMB\033[0m\n",
             keyframe_count_+1, std_manager_->data_base_.size(), total_stds, db_mem/(1024.*1024.),
             std_manager_->plane_cloud_vec_.size(), total_planes, plane_mem/(1024.*1024.),
             (db_mem+plane_mem)/(1024.*1024.));
    }

    std_manager_->AddSTDescs(stds_vec);
    keyframe_count_++;
  }

private:

  // ===== ICP 精化变换（参考 VoxelSLAM 的 icp_normal）=====
  // 不再返回 bool，而是输出最小特征值和收敛状态，让调用方决策
  void icpRefine(
    pcl::PointCloud<pcl::PointXYZINormal>& pl_src,
    pcl::PointCloud<pcl::PointXYZINormal>& pl_tar,
    std::pair<Eigen::Vector3d, Eigen::Matrix3d>& pose,
    double& out_min_eig, bool& out_converge)
  {
    out_min_eig = 0;
    out_converge = false;

    if (pl_src.empty() || pl_tar.empty()) {
      printf("[ICP] Empty clouds: src=%zu tar=%zu\n", pl_src.size(), pl_tar.size());
      return;
    }

    pcl::KdTreeFLANN<pcl::PointXYZ> kd_tree;
    pcl::PointCloud<pcl::PointXYZ>::Ptr tar_xyz(new pcl::PointCloud<pcl::PointXYZ>());
    for (auto& ap : pl_tar.points) {
      pcl::PointXYZ pi; pi.x=ap.x; pi.y=ap.y; pi.z=ap.z;
      tar_xyz->push_back(pi);
    }
    kd_tree.setInputCloud(tar_xyz);

    std::vector<int> idx(1);
    std::vector<float> dist(1);
    Eigen::Vector4d paras(0.2, 0.2, 0.5, 3); // 先松
    int is_converge = 0;
    int ssize = pl_src.size();
    Eigen::Matrix3d mat_norm;
    int total_match = 0;

    for (int iter = 0; iter < 20; iter++)
    {
      Eigen::Matrix<double,6,6> Hess = Eigen::Matrix<double,6,6>::Zero();
      Eigen::Matrix<double,6,1> JacT = Eigen::Matrix<double,6,1>::Zero();
      int match_num = 0;
      mat_norm.setZero();

      for (int i = 0; i < ssize; i++)
      {
        auto& sp = pl_src[i];
        Eigen::Vector3d plocal(sp.x, sp.y, sp.z);
        Eigen::Vector3d pi = pose.second * plocal + pose.first;
        pcl::PointXYZ search_pt;
        search_pt.x=pi[0]; search_pt.y=pi[1]; search_pt.z=pi[2];

        Eigen::Vector3d ni(sp.normal_x, sp.normal_y, sp.normal_z);
        ni = pose.second * ni;

        if (kd_tree.nearestKSearch(search_pt, 1, idx, dist) > 0) {
          auto& np = pl_tar[idx[0]];
          Eigen::Vector3d tpi(np.x, np.y, np.z);
          Eigen::Vector3d tni(np.normal_x, np.normal_y, np.normal_z);
          Eigen::Vector3d ninc = ni - tni, nadd = ni + tni;
          double pt2pt = (pi - tpi).norm();
          double pt2pl = fabs(tni.transpose() * (pi - tpi));

          if ((ninc.norm() < paras[0] || nadd.norm() < paras[1]) &&
              pt2pl < paras[2] && pt2pt < paras[3]) {
            double rr = tni.dot(pi - tpi);
            Eigen::Matrix3d phat;
            phat << 0,-plocal(2),plocal(1), plocal(2),0,-plocal(0), -plocal(1),plocal(0),0;
            Eigen::Matrix<double,6,1> jac;
            jac.head(3) = phat * pose.second.transpose() * tni;
            jac.tail(3) = tni;
            Hess += jac * jac.transpose();
            JacT += jac * rr;
            match_num++;
            mat_norm += tni * tni.transpose();
          }
        }
      }

      total_match = match_num;
      if (match_num < 3) break;  // 放松到3（原来5）

      Eigen::Matrix<double,6,1> dxi = Hess.ldlt().solve(-JacT);
      double ang = dxi.head(3).norm();
      if (ang > 1e-11) {
        Eigen::Vector3d axis = dxi.head(3) / ang;
        Eigen::Matrix3d K;
        K << 0,-axis(2),axis(1), axis(2),0,-axis(0), -axis(1),axis(0),0;
        pose.second = pose.second * (Eigen::Matrix3d::Identity() + sin(ang)*K + (1-cos(ang))*K*K);
      }
      pose.first += dxi.tail(3);

      if (dxi.head(3).norm() < 1e-3 && dxi.tail(3).norm() < 1e-3) {
        if (is_converge) break;
        paras << 0.1, 0.1, 0.1, 1;  // 收紧
        is_converge = 1;
      }
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(mat_norm);
    Eigen::Vector3d eig = saes.eigenvalues();
    out_min_eig = eig[0];
    out_converge = (is_converge == 1);

    printf("[ICP] matches=%d eig=(%.2f,%.2f,%.2f) converge=%d\n",
           total_match, eig[0], eig[1], eig[2], is_converge);
  }

  struct FrameData {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
    Eigen::Matrix3d R; Eigen::Vector3d p; int id;
  };

  struct KeyframeInfo {
    Eigen::Vector3d pos;
    double journey;
    int btc_id;          // ★ BTC 顺序编号（和 SearchLoop 返回值对应）
    int lio_frame_id;    // LIO 帧号（仅用于打印）
  };

  LoopDetectorConfig cfg_;
  STDescManager* std_manager_;
  std::deque<FrameData> frame_buffer_;
  std::vector<KeyframeInfo> keyframe_infos_;
  int frame_count_, keyframe_count_;
  int cooldown_;
  double total_journey_ = 0;
  Eigen::Vector3d last_key_pos_ = Eigen::Vector3d::Zero();
};

#endif